# Imports here
%matplotlib inline

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

# Load the Data
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Build and train
from collections import OrderedDict
from torch import nn
import torch.optim as optim

# Class Prediction
import predict


# image = mpimg.imread('flower_data/train/1/image_06734.jpg')
# plt.imshow(image

data_dir = './flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# TODO: Define your transforms for the training and validation sets
data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
#                                       transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.CenterCrop(32),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir,transform=data_transforms)
test_data = datasets.ImageFolder(valid_dir,transform=data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 980
# percentage of training set to use as validation
valid_size = 0.2

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# print(valid_loader.dataset)
# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

classes = []
for key in cat_to_name:
    classes.append(cat_to_name[key])
# classes

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

# TODO: Build and train your network
model = models.vgg16(pretrained = True)
# freeze all VGG parameters since we're only optimizing the target image
for param in model.parameters():
    param.requires_grad_(False)

# Create CNN model
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (1024 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

# create a complete CNN
model = Net()
print(model)

# Create classifier using Sequential with OrderedDict
# classifier = nn.Sequential(OrderedDict([
#                             ('conv1', nn.Conv2d(3, 16, 3, padding=1)),
#                             ('relu', nn.ReLU()),
#                             ('pool', nn.MaxPool2d(2, 2)),
#                             ('conv2', nn.Conv2d(16, 32, 3, padding=1)),
#                             ('relu', nn.ReLU()),
#                             ('pool', nn.MaxPool2d(2, 2)),
#                             ('conv3', nn.Conv2d(32, 64, 3, padding=1)),
#                             ('relu', nn.ReLU()),
#                             ('pool', nn.MaxPool2d(2, 2)),

#                             ('dropout', nn.Dropout(0.25)),
#                             ('fc1', nn.Linear(1024, 512)),
#                             ('relu', nn.ReLU()),
#                             ('dropout', nn.Dropout(0.25)),
#                             ('fc2', nn.Linear(512, 102)),

#                             ('output', nn.LogSoftmax(dim=1))
#                              ]))

model.classifier = classifier

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

# move the model to GPU, if available
device = torch.device("cuda" if train_on_gpu else "cpu")
model.to(device)

## Train the Network
# number of epochs to train the model
n_epochs = 30

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_imgclassifier.pt')
        valid_loss_min = valid_loss

# TODO: Save the checkpoint
model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {'model_state': model.state_dict(),
              'criterion_state': criterion.state_dict(),
              'optimizer_state': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'epochs': epochs,
              'best_train_loss': train_loss,
              # 'Best train accuracy': epoch_train_accuracy,
              'best_validation_loss': valid_loss,
              # 'Best Validation accuracy': epoch_val_acc
              }
torch.save(checkpoint, 'model_imgclassifier.pt')

# TODO: Write a function that loads a checkpoint and rebuilds the model
checkpoint = torch.load('model_imgclassifier.pt')

model.load_state_dict(checkpoint['model_state'])
criterion.load_state_dict(checkpoint['criterion_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
image_datasets['train'] = load_state_dict(checkpoint['class_to_idx'])
epoch = checkpoint['epochs']
train_loss = checkpoint['best_train_loss']
valid_loss = checkpoint['best_validation_loss']


# TODO: Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    max_size = 256
    shape=None

    img_pil = Image.open(image).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape

    img_transforms = transforms.Compose([transforms.Resize(size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)

    return image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    labels = cat_to_name.json
    gpu_available = torch.cuda.is_available()
    probs, classes = predict.predict(image=image_path, checkpoint=model, labels=labels, gpu=gpu_available)
    return probs, classes


topk_probs, topk_classes = predict(image='sample_img.jpg', checkpoint='my_model.pt')
label = topk_probs[0]
prob = topk_classes[0]

print(f'Flower      : {cat_to_name[label]}')
print(f'Label       : {label}')
print(f'Probability : {prob*100:.2f}%')

print(f'\nTop K\n---------------------------------')

for i in range(len(top_prob)):
    print(f"{cat_to_name[top_classes[i]]:<25} {top_prob[i]*100:.2f}%")
