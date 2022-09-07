'''
Author: Michael Xu

Deep Q-learning (without experience replay)

In Deep Q-Learning, it is assumed that our state x action space is too large,
so we need to find a function approximator for Q(s, a), which is usually 
a neural network.


'''
import gym
from envs.grid_world import GridWorldEnv
import numpy as np
import random
import torch
from torch import nn

# environment parameters
grid_size = 5

# training parameters
eps = 0.1       # exploration likelihood
gamma = 0.99    # discount factor
alpha = 0.099     # learning rate


# Set up Q(s, a)
# state space S size is (n,n,n,n), which is (n,n) for agent position and (n,n) for target position
# action space A size 4, so the Q space is (n,n,n,n,4)
#Q = np.zeros(shape=(grid_size, grid_size, grid_size, grid_size, 4))

# (x,y) for agent, (x,y) for target, and a for action
Q = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))


def simulate(steps=1000,train=False):
    global Q
    
    # 0. initialize environment
    mode = None if train else "human"
    env = GridWorldEnv(size=grid_size,render_mode=mode)
    env.action_space.seed(42)
    

    # 1. Initialize Q(s, a) for all states and actions
    # initialized theta = 0
    
    # 2. sample initial state s
    observation = env.reset(seed=42)

    # 8. Repeat 3 - 7 for # of training steps
    for _ in range(steps):
        agent_pos = observation["agent"]
        target_pos = observation["target"]

        # 3. Sample action a (epsilon-greedy)
        action = 0
        greedy = random.random() > eps
        if greedy:
            # argmax for best action used in greedy update
            possible_returns = []
            for action in range(env.action_space.n):
                Q_val = Q.forward(np.concatenate((agent_pos, target_pos, action), axis=None))
                possible_returns.append(Q_val)
            possible_returns = np.array(possible_returns)

            # Get all the actions that return the best value, then randomly select one
            # If there is only one best value, then this is deterministic
            best_action = np.random.choice(np.flatnonzero(possible_returns == possible_returns.max()))
            action = best_action
        else:
            action = env.action_space.sample()

        index = tuple(np.concatenate((agent_pos, target_pos, action), axis=None))
        
        # 4. Take a step through the environment, and get next state s', reward, done signal
        observation, reward, done, info = env.step(action)
        
        estimate = 0
        if done: # 5a. If done: estimate of Q*(s, a) = reward, reset initial state s
            estimate = reward
            observation = env.reset()
        else: # 5b. Else: estimate of Q*(s, a) = reward + discounted max(Q(s', ))
            new_agent_pos = observation["agent"]
            new_target_pos = observation["target"]

            # argmax for best action of next state
            possible_returns = []
            for action in range(env.action_space.n):
                Q_val = Q.forward(np.concatenate((new_agent_pos, new_target_pos, action), axis=None))
                possible_returns.append(Q_val)
            possible_returns = np.array(possible_returns)

            estimate = reward + gamma * possible_returns.max()

        # 6. Update Q-network
        if train:
            Q[index] = (1.0 - alpha) * Q[index] + alpha * estimate

        if env.render_mode == "human":
            env.render()
    env.close()

print("training...")
simulate(steps = 100000, train=True)
print("testing...")
eps = 0 # don't explore when testing
simulate(steps = 1000, train=False)