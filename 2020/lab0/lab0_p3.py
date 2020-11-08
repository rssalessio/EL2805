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

### CREATE RL ENVIRONMENT ###
env = gym.make('CartPole-v0')        # Create a CartPole environment
n = len(env.observation_space.low)   # State space dimensionality
m = env.action_space.n               # Number of actions

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
        action  = np.random.randint(m)   # Pick a random integer between
                                         # [0, m-1]


        # The next line takes permits you to take an action in the RL environment
        # env.step(action) returns 4 variables:
        # (1) next state; (2) reward; (3) done variable; (4) additional stuff
        next_state, reward, done, _ = env.step(action)

        state = next_state

# Close all the windows
env.close()
