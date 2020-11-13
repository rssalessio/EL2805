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
# Problem: Dueling DQN (6.10)
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 12th November 2020, by alessior@kth.se
#

# Import PyTorch libraries
import torch
import torch.nn as nn

# Parameters of the problem, freely chosen
m = 2  # Number of actions
d = 2  # State dimensionality

# Network class, with one hidden layer of neurons.
# We will compute the output of the hidden layer given the state s.
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden = 16
        self.layer1 = nn.Linear(state_dim, hidden)
        # Value layer
        self.value = nn.Linear(hidden, 1)
        # Advantage layer
        self.adv = nn.Linear(hidden, action_dim)
        self.layer1_activation = nn.ReLU()

    def forward(self, x):
        y1 =  self.layer1(x)
        y1 = self.layer1_activation(y1)
        # Compute advantage
        value = self.value(y1)
        # Compute value
        adv = self.adv(y1)
        # Enforce constraint
        advAverage = torch.mean(adv, dim=1, keepdim=True)
        return value + adv - advAverage


# Example
net = DQNetwork(d, m)
x = torch.tensor([[1.] * d])    # Create a batch of data of dimension 1xd
print("Input: {} - Output: {}".format(
    x.detach().numpy(), net(x).detach().numpy()))
