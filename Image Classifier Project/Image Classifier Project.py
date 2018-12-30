# Imports here
%matplotlib inline

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import transforms, models

# Load the Data
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Build and train
from collections import OrderedDict
from torch import nn
import torch.optim as optim

# Sanity Checking
import seaborn as sns


# Load the data
data_dir = './flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# TODO: Define your transforms for the training and validation sets
data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30),
                                 transforms.RandomResizedCrop(224),
#                                  transforms.CenterCrop(32),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]),
    
    'valid': transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.CenterCrop(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    
    'test': transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.CenterCrop(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
}

# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train' : datasets.ImageFolder(train_dir,transform=data_transforms['train']),
    'valid' : datasets.ImageFolder(valid_dir,transform=data_transforms['valid'])
}

# TODO: Using the image datasets and the trainforms, define the dataloaders

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 30
# percentage of training set to use as validation
valid_size = 0.2

# obtain training indices that will be used for validation
num_train = len(image_datasets['train'])
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
dataloaders = {
        'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, sampler=train_sampler, num_workers=num_workers),
        'valid' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers),
        'test' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size, num_workers=num_workers)
}

# Label mapping
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Building and training the classifier
# TODO: Build and train your network
model = models.vgg19(pretrained = True)
# freeze all VGG parameters since we're only optimizing the target image
for param in model.parameters():
    param.requires_grad_(False)
model.classifier

# Create classifier using Sequential with OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1024)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(1024, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier
model.classifier

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    model.cuda()

# move the model to GPU, if available
device = torch.device("cuda" if train_on_gpu else "cpu")
# model.to(device)

## Train the Network
# number of epochs to train the model
n_epochs = 2 #30

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    train_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    train_correct = 0.0
    for data, target in dataloaders['train']:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        _, preds = torch.max(output,1)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        train_correct += torch.sum(preds == target.data)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    validate_correct = 0.0
    for data, target in dataloaders['valid']:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        equal = (output.max(dim=1)[1] == target.data)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        validate_correct += torch.sum(equal)#type(torch.FloatTensor)
    
    # calculate average losses
    train_loss = train_loss/len(dataloaders['train'].dataset)
    train_acc = train_correct.double()/len(dataloaders['train'].dataset)
    valid_loss = valid_loss/len(dataloaders['valid'].dataset)
    valid_acc = validate_correct.double()/len(dataloaders['valid'].dataset)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tAcc: {:.6f} \n\t\tValidation Loss: {:.6f} \tAcc: {:.6f}'.format(
        epoch, train_loss, train_acc, valid_loss, valid_acc))
    
    # TODO: Save the checkpoint 
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        
        model.class_to_idx = image_datasets['train'].class_to_idx

        checkpoint = {'arch': 'vgg19',
              'model_state': model.state_dict(),
              'criterion_state': criterion.state_dict(),
              'optimizer_state': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'epochs': n_epochs,
              'best_train_loss': train_loss,
              # 'Best train accuracy': epoch_train_accuracy,
              'best_validation_loss': valid_loss,
              # 'Best Validation accuracy': epoch_val_acc
              }
        torch.save(checkpoint, 'model_imgclassifier.pt')
        
        
        valid_loss_min = valid_loss

# Save the checkpoint
# Done above

# Loading the checkpoint
# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_model(checkpoint_path):
    checkpoint = torch.load('model_imgclassifier.pt')
    
    if checkpoint['arch'] == 'vgg19':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad_(False) 
        
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1024)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(1024, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
        model.classifier = classifier

        model.load_state_dict(checkpoint['model_state'])
        criterion.load_state_dict(checkpoint['criterion_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        model.class_to_idx = checkpoint['class_to_idx']
        n_epochs = checkpoint['epochs']
        train_loss = checkpoint['best_train_loss']
        valid_loss = checkpoint['best_validation_loss']
    
    return model


model_load = load_model('model_imgclassifier.pt')

# Image Preprocessing
# TODO: Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Open the image
    from PIL import Image
    img = Image.open(image)
    
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    if title:
        plt.title(title)
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
#     image = image.numpy().transpose((1, 2, 0))
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

_= imshow(process_image('image_sample.jpg'))

# Class Prediction
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labels, top_flowers

# Sanity Checking
# TODO: Display an image along with the top 5 classes
def plot_solution(image_path, model):
    # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    # Set up title
#     flower_num = image_path.split('/')[1]
#     title_ = label_map[flower_num]

    # Plot flower
    img = process_image(image_path)
    imshow(img, ax);
    
    # Make prediction
    probs, labs, flowers = predict(image_path, model) 
    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    plt.show()

plot_solution('image_sample.jpg', model_load)