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


data_dir = '/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# TODO: Define your transforms for the training and validation sets
data_transforms = transforms.Compose([
	transforms.RandomRotation(30),
	transforms.RandomHorizontalFlip(p=0.5),
	transforms.CenterCrop(10),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
test_data = datasets.ImageFolder(valid_dir, transform=data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 30
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


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

classes = [] ''' need to be implemented'''

# TODO: Build and train your network
model = models.vgg16(pretrained = True)

# Create classifier using Sequential with OrderedDict
classifier = nn.Sequential(OrderedDict([
							('conv1', nn.Conv2d(3, 16, 3, padding=1)),
                            ('relu', nn.ReLU()),
                            ('pool', nn.MaxPool2d(2, 2)),
          					('conv2', nn.Conv2d(16, 32, 3, padding=1)),
          					('relu', nn.ReLU()),
          					('pool', nn.MaxPool2d(2, 2)),
          					('conv3', nn.Conv2d(32, 64, 3, padding=1)),
          					('relu', nn.ReLU()),
          					('pool', nn.MaxPool2d(2, 2)),

                            ('dropout', nn.Dropout(0.25)),
                            ('fc1', nn.Linear(1024, 512)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(0.25)),
                            ('fc2', nn.Linear(512, 102)),

                            ('output', nn.LogSoftmax(dim=1))
                             ]))

model.classifier = classifier

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)





# TODO: Save the checkpoint


# TODO: Write a function that loads a checkpoint and rebuilds the model