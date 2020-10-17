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
# Course: EL2805 - Reinforcement Learning - Lab 0 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 17th October 2020, by alessior@kth.se

### IMPORT PACKAGES ###
# numpy for numerical/random operations
# gym for the Reinforcement Learning environment
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pdb

class ExperienceReplayBuffer(object):
    def __init__(self, maximum_length=1000):
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n):
        if n > len(self.buffer):
            print('Error! Asked to retrieve too many elements from the buffer')

        indices = np.random.choice(len(self.buffer), n, replace=False)

        batch = [self.buffer[i] for i in indices]

        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

class MyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        # self.input_layer = nn.Linear(input_size, 8)
        # self.input_layer_activation = nn.ReLU()
        # self.output_layer = nn.Linear(8, output_size)
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, output_size),
        )

    def forward(self, x):
        # l1 = self.input_layer(x)
        # l1 = self.input_layer_activation(l1)
        # return self.output_layer(l1)
        return self.network(x)

### CREATE RL ENVIRONMENT ###
env = gym.make('CartPole-v0')        # Create a CartPole environment
n = len(env.observation_space.low)   # State space dimensionality
m = env.action_space.n               # Number of actions
buffer = ExperienceReplayBuffer(maximum_length=50)
network = MyNetwork(input_size=n, output_size=m)
optimizer = optim.Adam(network.parameters(), lr=0.0001)

### PLAY ENVIRONMENT ###
# The next while loop plays 5 episode of the environment
for episode in range(5):
    state = env.reset()                  # Reset environment, returns initial
                                         # state
    done = False                         # Boolean variable used to indicate if
                                         # an episode terminated

    while not done:
        env.render()                     # Render the environment
                                         # (DO NOT USE during training of the
                                         # labs...)
        # action  = np.random.randint(m)   # Pick a random integer between
                                         # [0, m-1]
        # pdb.set_trace()
        state_tensor =  torch.tensor([state],
            requires_grad=False, dtype=torch.float32)
        action_values = network(state_tensor)
        val, action = action_values.max(1)
        action = action.item()

        # The next line takes permits you to take an action in the RL environment
        # env.step(action) returns 4 variables:
        # (1) next state; (2) reward; (3) done variable; (4) additional stuff
        next_state, reward, done, _ = env.step(action)

        exp = (state, action, reward, next_state, done)
        buffer.append(exp)

        if len(buffer) >= 3:
            optimizer.zero_grad()
            states, actions, rewards, next_states, dones = buffer.sample_batch(3)
            states_values = network(
                torch.tensor(states, requires_grad=True, dtype=torch.float32))
            target_values = torch.zeros_like(states, requires_grade=False,
                            dtype=torch.float32)

            loss = nn.functional.mse_loss(states_values, target_values)
            loss.backward()
            nn.utils.clip_grad_norm_(neural network.parameters(), 1.)
            optimizer.step()

# Close all the windows
env.close()
