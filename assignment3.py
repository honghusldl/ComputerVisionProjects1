import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
class_names = ['T-shirt','Trouser','Sweater','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']

# Create data loaders
train_loader = DataLoader(train_data, batch_size=10, shuffle = True)
test_loader = DataLoader(test_data, batch_size=10, shuffle = True)

# Examine a batch of images
# Use DataLoader, make_grid and matplotlib to display the first batch of 10 images.
# display the labels as well

images, labels = next(iter(train_loader))
img_grid = make_grid(images, nrow = 10)
fig, ax = plt.subplots()
plt.imshow(img_grid.permute(1,2,0))
ax.set_xlabel('Image Label')
ax.set_xticks([14+30*x for x in range(0,10)])
ax.set_xticklabels(labels.tolist())
ax.get_yaxis().set_ticks([])
plt.show()

# # display first batch of 10 images
# batch_size = 10
# def show_images(images,nmax = batch_size):
#     fig, ax = plt.subplots(figsize = (10,10))
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.imshow(make_grid((images.detach()[:nmax]), nrow=5).permute(1,2,0)) # permute() put channels as the last dimension
#
# def show_batch(dataloader, nmax = batch_size):
#     for images in dataloader:
#         show_images(images,nmax)
#         break
# show_batch(train_loader)


# Downsampling
# If a 28x28 image is passed through a Convolutional layer using a 5x5 filter, a step size of 1, and no padding,
# create the conv layer and pass in one data sample as input, then printout the resulting matrix size
conv1 = nn.Conv2d(1,1,5,1) # nn.Conv2d(1,1,(5,5),1)
sample_x = train_data[0][0]
print(sample_x.shape)

# convert sample x to 4D
sample_x = sample_x.view(1,1,28,28)

x = F.relu(conv1(sample_x))
print(x.shape)

# If the sample from question 3 is then passed through a 2x2 MaxPooling layer
# create the pooling layer and pass in one data sample as input, then printout the resulting matrix size
x = F.max_pool2d(x,2) # stride length is not specified
print(x.shape)

conv1 = nn.Conv2d(1, 5, 2, 1)
conv2 = nn.Conv2d(5, 15, 2, 1)
x = train_data[1][0]
x = x.view(1,1,28,28)
x = F.relu(conv1(x))
print(x.shape)
x = F.max_pool2d(x,2,2)
print(x.shape)
x = F.relu(conv2(x))
print(x.shape)
x = F.max_pool2d(x,2,2)
print(x.shape)
x = x.view(-1,6*6*15)
print(x.shape)

# Define a convolutional neural network
# Define a CNN model that can be trained on the Fashion-MNIST dataset.
# The model should contain two convolutional layers, two pooling layers, and two fully connected layers.
# You can use any number of neurons per layer so long as the model takes in a 28x28 image and returns an output of 10.
# and then printout the count of parameters of your model

# input -> conv1 -> Relu -> Pooling1 -> conv2 -> Relu -> Pooling2 -> fc1 -> Relu -> fc2 -> sofrmax -> output
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,5,2,1)
        self.conv2 = nn.Conv2d(5,15,2,1)

        self.fc1 = nn.Linear(6*6*15,150)
        self.fc2 = nn.Linear(150, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(-1, 6*6*15)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim = 1)
        return x

model = CNN()
print(model)

# Define loss function & optimizer
# Define a loss function called "criterion" and an optimizer called "optimizer".
# You can use any loss functions and optimizer you want,
# although we used Cross Entropy Loss and Adam (learning rate of 0.001) respectively.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

# Train and test the model
# try with any epochs you want
# and printout some interim results
epochs = 10
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    for b, (x_train, y_train) in enumerate(train_loader):
        b += 1

        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred, 1)[1]
        batch_corr = (predicted ==y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 3000 == 0:
            print(f'epoch:{i} batch:{b} loss: {loss.item()} accuracy: {trn_corr.item() * 100 / (10 * b)}%')

    # run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)

            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()
    print(f'Test accuracy: {tst_corr.item() * 100 / (len(test_data))}%')

# epoch 9 accuracy: 91.8%
# test accuracy 89.4%

# Remember, always experiment with different architecture and different hyper-parameters, such as
# different activation function, different loss function, different optimizer with different learning rate
# different size of convolutional kernels, and different combination of convolutional layers/pooling layers/FC layers
# to make the best combination for solving your problem in real world

