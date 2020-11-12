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
# Course: EL2805 - Reinforcement Learning
# Problem: Design of Q networks (6.6 part 2)
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 12th November 2020, by alessior@kth.se
#

# Import PyTorch libraries
import torch
import torch.nn as nn

# Parameters of the problem, freely chosen

# Parameters of the problem, freely chosen
K = 1  # Bound on the Q value
m = 1  # Action dimensionality
d = 2  # State dimensionality

# Network class, with one hidden layer of neurons.
# We will compute the output of the hidden layer given the state s. Then
# we will concatenate the output of the hidden layer with the action.
# The output layer will take as input this new concatenated vector
class Network2(nn.Module):
    def __init__(self, d, m, K):
        super().__init__()
        hidden_neurons = 10  # Number of hidden neurons
        # The input dimensionality of the network should be equal to the
        # dimensionality of the state
        self.input_state_layer = nn.Linear(d, hidden_neurons)
        # The dimensionality of the input of the next layer should be equal to
        # the dimensionality of the hidden layer + dimensionality of the actions
        # The output should be equal to 1 since we are computing just one Q value
        self.output_layer = nn.Linear(hidden_neurons + m, 1)
        # Use the Tanh activation to bound the output between -1 and 1
        self.output_activation = nn.Tanh()
        self.K = K

    def forward(self, s, a):
        # Computation of Q(s,a)

        # Compute the hidden layer
        h_state = self.input_state_layer(s)
        # Concatenate output of the hidden layer with the action along
        # the dimensionality of the data
        hidden = torch.cat([h_state, a], dim=1)
        # Compute ouput
        out = self.output_layer(hidden)
        # Multiply the output of the activation function by K to get that the
        # output is between -K and K
        return self.K * self.output_activation(out)


# Example
net = Network2(d, m, K)
x = torch.tensor([[1.] * d])    # Create a batch of data of dimension 1xd
a = torch.tensor([[0.5] * m])       # Create a batch of data of dimension 1xm
print("Input: {}/{} - Output: {}".format(
    x.detach().numpy(), a.item(), net(x, a).detach().numpy()))
