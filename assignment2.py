# Import torch and NumPy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F          # adds some efficiency
# from torch.utils.data import DataLoader  # lets us load data in batches
# from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Set the random seed for NumPy and PyTorch both to "42"
#   This allows us to share the same "random" results.
torch.manual_seed(42)
np.random.seed(42)

# Create a NumPy array called "arr" that contains 6 random integers between 0 (inclusive) and 5 (exclusive)
arr = np.random.randint(0,5,6)
print(arr)
# Create a tensor "x" from the array above
x = torch.from_numpy(arr)
print(x)

# Change the dtype of x from 'int32' to 'int64'
#arr.astype("int64")
x = torch.tensor(arr,dtype = torch.int64)
print(x)

# Reshape x into a 3x2 tensor
# x = x.reshape(3,2)
x = x.view(3,-1)
print(x)

# Return the left-hand column of tensor x
print(x[:,0].reshape(3,1))

# Without changing x, return a tensor of square values of x
print(torch.square_(x))

# Create a tensor "y" with the same number of elements as x, that can be matrix-multiplied with x
y = torch.randint(0,5,[2,3]) # 6 elements
print(y)

# Find the matrix product of x and y
print(torch.mm(x,y))

# Create a Simple linear model using torch.nn
# the model will take 1000 input and output 20 multi-class classification results.
# the model will have 3 hidden layers which include 200, 120, 60 respectively.

# model = nn.Linear(in_features=1000, out_features=20)
# Define the ANN model
class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_features = 1000, out_features = 20, layers = [200,120,60]):
        super().__init__()
        # self.fc1 = nn.Linear(1000, 200)
        # self.fc2 = nn.Linear(200, 120)
        # self.fc3 = nn.Linear(120,60)
        # self.fc4 = nn.Linear(60, 20)
        self.fc1 = nn.Linear(in_features,layers[0])
        self.fc2 = nn.Linear(layers[0],layers[1])
        self.fc3 = nn.Linear(layers[1],layers[2])
        self.fc4 = nn.Linear(layers[2],out_features)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim = 1)
        return x

# initiate the model and printout the number of parameters
model = MultiLayerPerceptron()
print(model)

for param in model.parameters():
    print(param.numel())