# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - PyTorch Tutorial
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 25th October 2020, by alessior@kth.se

# Start by importing the basic libraries
import numpy as np               # Numerical computing
import matplotlib.pyplot as plt  # Plotting utilities
import torch                     # Basic Pytorch import
import torch.nn as nn            # Imports neural networks blocks
from ipywidgets import interact, IntSlider, FloatSlider


# ### 1. Input/Target output data
#
# Define tensors for both the input data and the target output data.
#
# To create a tensor use the notation
# ```torch.tensor(data, dtype=torch.float64)```
# (https://pytorch.org/docs/stable/generated/torch.tensor.html)
#
# - Make sure to use ```dtype=torch.float64```. This forces Pytorch to use double precision
# - ```requires_grad=False``` tells Pytorch to consider this variable as a parameter.
# Therefore it is not possible to compute the gradient with respect to this variable.
# - The data passed to ```torch.tensor()``` can be a list or numpy array.


# Input (temp, rainfall, humidity)
inputs = torch.tensor([[73, 67, 43],
                       [91, 88, 64],
                       [87, 134, 58],
                       [102, 43, 37],
                       [69, 96, 70]],
                       dtype=torch.float32,
                       requires_grad=False)

# Target outputs (apples, oranges)
target_outputs = torch.tensor([[56, 70],
                               [81, 101],
                               [119, 133],
                               [22, 37],
                               [103, 119]],
                               dtype=torch.float32,
                               requires_grad=False)


# ### 2. Create the neural network
# ![alt text](nn.png "Data")
#
# Define a Neural network with 2 layers (input and output). The number of
# neurons should be a parameter of the class. Notice that we expect a positive
# output from the network  (hint: *is the target output positive or negative*?),
# therefore we should use an activation function that gives a positive output.
# We will use ReLU for the first layer as activation function.
#
# Remember that
# - ReLU activation : $\sigma(x) = \max(0,x)$
# - Sigmoidal activation: $\sigma(x) = \frac{1}{1+e^{-x}}$
#


class NeuralNet(nn.Module):
    # In the init define each layer individually
    # \par `neurons` is used to define the number of input neurons
    # by default we assign 3
    def __init__(self, neurons=3):
        super().__init__()  # This line needs to called to properly setup the network
        self.linear1 = nn.Linear(3, neurons) # Layer with 3 inputs and `neurons` output
        self.act1 = nn.ReLU() # Activation function
        self.linear2 = nn.Linear(neurons, 2) # Layer with `neurons` inputs and 2 outputs

    # In the forward function you define how each layer is
    # interconnected. Observe that 'x' is the input.
    def forward(self, x):
        # First layer (input layer)
        x = self.linear1(x)
        x = self.act1(x)
        # Second layer (output)
        x = self.linear2(x)
        return x

# Pay attention, there is no need to define "backward" for the backward
# pass. It is done by the library


# ### 3. Loss function


# Define a utility function to train the model
def fit(neurons, learning_rate, momentum, nesterov, plot=True):
    # Initialize network and optimizer
    model = NeuralNet(neurons)
    # Define SGD optimizer with the parameters of the network and the
    # given parameters (learning rate, momentum and nesterov)
    opt = torch.optim.SGD(model.parameters(), learning_rate,
        momentum=momentum, nesterov=nesterov)
    losses = []
    # Train for 1000 epochs
    for epoch in range(1000):
        # Reset gradients
        opt.zero_grad()
        # Generate predictions
        pred = model(inputs)
        # Compute MSE loss
        loss = nn.functional.mse_loss(pred, target_outputs)
        # Append loss to loss vector (used for plotting)
        losses.append(loss.item())
        # Compute gradients
        loss.backward()
        # Compute backward step
        opt.step()
    if plot:
        plt.plot(losses)
        plt.grid(alpha=0.5)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        plt.show()
    return model


# Try model with 6 neurons, learning rate of 10^-6, momentum of 0.9 and no Nesterov
model = fit(6, 1e-6, 0.9, False, plot=True);
pred = model(inputs)
print('Prediction: {}'.format(pred.detach().numpy()))
print('Real values: {}'.format(target_outputs))
