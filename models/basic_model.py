from __future__ import print_function, division
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# based on pytorch blitz tutorial https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

#--------- Training params  ------------#

BATCH_SIZE = 30
NUM_EPOCHS = 1

#--------- Load and transform data ------------#

transform = transforms.Compose(
    [transforms.ToTensor(),  # convert PIL Image to tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # normalize RBG image
)

cifar_directory = './data/CIFAR/'

trainset = torchvision.datasets.CIFAR10(os.path.join(cifar_directory, 'train'),
                                                 train=True,
                                                 transform=transform,
                                                 download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(os.path.join(cifar_directory, 'test'),
                                                 train=False,
                                                 transform=transform,
                                                 download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#--------- Visualize some images ------------#

visualize = False

def imshow(img):
    img = img/2 + 0.5 # unnormalize
    npimg = img.numpy() # convert from tensor to numpy array
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

if visualize:
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join(classes[labels[j]] for j in range(BATCH_SIZE)))


#--------- Define neural network  ------------#

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input is nSamples x nChannels x Height x Width
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 input image channel, 6 output channels, 5x5 square convolution kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)  # y = Wx + b
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pool over a (2, 2) window
        x = self.pool(F.relu(self.conv1(x)))
        # max pool over a square
        x = self.pool(F.relu(self.conv2(x)))
        # flatten
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

#--------- Define loss and optimmizer  ------------#

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001, momentum=0.9)

# training loop:
# optimizer.zero_grad()
# output = net(input)
# loss = criterion(output, target)
# loss.backward() # backprop gradients
# optimizer.step()


#--------- Train network  ------------#

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data # batch of inputs

        # zero param gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print stats
        running_loss += loss.item() # get python number value from tensor
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('Training complete')

#--------- Evaluate network on test data ------------#

testdataiter = iter(testloader)
images, labels = testdataiter.next()

outputs = net(images)
print(outputs.size())
_, predicted = torch.max(outputs, 1) # returns [values, indices]

print('Ground truth: ', " ".join("%5s" % classes[labels[j]] for j in range(BATCH_SIZE)))
print('Predicted: ', " ".join("%5s" % classes[predicted[j]] for j in range(BATCH_SIZE)))


#--------- Set up to run on GPU  ------------#

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
