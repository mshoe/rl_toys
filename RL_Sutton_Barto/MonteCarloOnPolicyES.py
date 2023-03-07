import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

class MontoCarloOnPolicyES():

    def __init__(self, env):
        self.env = env

        # Assume the observation space is a spaces.Dict, and action space is spaces.Discrete. 
        # In the future, we will want to handle all gym spaces

        self.total_obs_space = np.array([], dtype=np.int)
        for key in env.observation_space:
            self.total_obs_space = np.append(self.total_obs_space, env.observation_space[key].high - env.observation_space[key].low + 1)
        
        self.total_state_action_space = np.append(self.total_obs_space, env.action_space.n)

        self.Q = np.zeros(self.total_state_action_space)
        self.num_returns = np.zeros(self.total_state_action_space) # store the number of returns sampled for each state action pair

        self.policy = np.random.randint(0, env.action_space.n, size=self.total_obs_space)
        # construct the pdeterministic policy as a numpy tensor based on the environment's obersevation space
        # constuct the Q function as a numpy tensor based on the environment's observation space and action space
        


        pass

    def learn(self, gamma, num_episodes, _max_episode_length=100, verbose=False):

        curr_episode = 0
        for episode in range(num_episodes):
            
            
            print("episode = ", episode)
            # generate an episode following the current policy
            episode = self._generate_episode(_max_episode_length, verbose)
            discounted_return = 0.0
            for step in episode:
                state = self._convert_obs_to_array(step[0])
                action = step[1]
                reward = step[2]
                discounted_return = gamma * discounted_return + reward

                

                self.num_returns[tuple(np.append(state, action))] += 1

                # update the Q function
                self.Q[tuple(np.append(state, action))] += (discounted_return - self.Q[tuple(np.append(state, action))]) / self.num_returns[tuple(np.append(state, action))]

                # update the policy
                self.policy[tuple(state)] = np.argmax(self.Q[tuple(state)])

    def _generate_episode(self, max_episode_length=100, verbose=False):
        # generate an episode following the current policy
        # return a list of (state, action, reward) tuples

        episode = []
        done = False

        obs, info = self.env.reset()

        # exploring starts, so we can start from any state
        obs = self.env.observation_space.sample()
        print("Start state =", self._convert_obs_to_array(obs))


        current_state_num = 0
        while not done and current_state_num < max_episode_length:
            current_state_num += 1
            state = self._convert_obs_to_array(obs)
            action = self.policy[tuple(state)]
            obs, reward, done, info = self.env.step(action)

            if verbose:
                print("***** step, ", current_state_num, "*****")
                print("state =",state)
                print("action =",self.env._action_to_acceleration[action])
                print("obs =",obs)
                print("info =",info)

            episode.append((obs, action, reward))

        if done:
            print("Episode done in ", current_state_num, " steps")

        if verbose:
            print("************episode done************")

        return episode
        

    def _convert_obs_to_array(self, obs):
        # convert the observation (dict) into numpy array
        state = np.array([], dtype=np.int)
        for key in self.env.observation_space:
            state = np.append(state, obs[key])
        return state


    # write a function that returns the value function from the Q function
    def get_value_function(self):
        value_function = np.zeros(self.total_obs_space)
        for state in np.ndindex(self.total_obs_space):
            value_function[state] = np.max(self.Q[tuple(np.append(state, np.arange(self.env.action_space.n)))])
        return value_function
    
