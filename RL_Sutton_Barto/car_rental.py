import matplotlib.pyplot as plt
import numpy as np


# This exercise is way more complicated than it seems, mainly because of the dynamics function

# in this code, we will implement policy iteration by iterative policy evaluation
# we will use numpy

MAX_NUM_CARS = 20

EXPECTED_RENTALS_LOC_1 = 3
EXPECTED_RENTALS_LOC_2 = 4

EXPECTED_RETURNS_LOC_1 = 3
EXPECTED_RETURNS_LOC_2 = 2

DISCOUNT_FACTOR = 0.9

# the action space is [-5, 5] which is the number of cars we can move between location 1 and 2,
# positive means moving from 1 to 2, negative means moving from 2 to 1
action_space = np.arange(start=-5, stop=6)

# the state space is a 2D vector, where each element can range from [0, 20]




# the number of cars requested and returned at each location is a poisson random variable
# where lambda is the expected number of cars requested/returned, and n is the number of cars
# requested/returned. We will use the following function to compute the probability of n cars
# requested/returned given lambda
def poisson(lamda, n):
    return (lamda ** n) * np.exp(-lamda) / np.math.factorial(n)


# 4 argument dynamics function
# returns a 2D matrix, where each element is the probability of the next state and the reward
# The next state depends on the poisson random variables

# IT is probably a good idea to write out the probability tables for transitions of rentals and returns,
# WITHOUT considering actions, just state.

def build_dynamics_table():
    transition_matrix = np.zeros(shape=(MAX_NUM_CARS+1, MAX_NUM_CARS+1, MAX_NUM_CARS+1, MAX_NUM_CARS+1, 2))

    for state_x in range(MAX_NUM_CARS+1):
        for state_y in range(MAX_NUM_CARS+1):

            # now compute the rent and returns
            # NOTE, THIS LOOP TAKES TOO LONG TO RUN, NEED TO COMPUTE THIS FASTER
            for rent_1 in range(0, MAX_NUM_CARS+1):
                for return_1 in range(0, MAX_NUM_CARS+1):
                    for rent_2 in range(0, MAX_NUM_CARS+1):
                        for return_2 in range(0, MAX_NUM_CARS+1):

                            # compute the probability of this rent and return
                            prob = poisson(EXPECTED_RENTALS_LOC_1, rent_1) * poisson(EXPECTED_RENTALS_LOC_2, rent_2) * poisson(EXPECTED_RETURNS_LOC_1, return_1) * poisson(EXPECTED_RETURNS_LOC_2, return_2)
                            
                            # compute the next state
                            post_state_x = min(max(0, state_x - rent_1 + return_1), 20)
                            post_state_y = min(max(0, state_y - rent_2 + return_2), 20)

                            # compute the reward (assume that returns happen after rent)
                            rent_reward_1 = 10 * min(rent_1, post_state_x)
                            rent_reward_2 = 10 * min(rent_2, post_state_y)

                            # the movement reward depends directly on the action, so it does not belong here
                            reward = rent_reward_1 + rent_reward_2

                            transition_matrix[state_x, state_y, post_state_x, post_state_y, 0] += prob # this sums to total probability of state transition
                            transition_matrix[state_x, state_y, post_state_x, post_state_y, 1] += prob * reward # this sums to expected reward

    return transition_matrix




#     return next_state, 

if __name__ == "__main__":
    # the policy is a 2D matrix, which represents the action [-5, 5] to take at each state
    policy = np.zeros(shape=(MAX_NUM_CARS+1, MAX_NUM_CARS+1))

    # the value function is a 2D matrix, which represents the value of each state
    value_function = np.zeros(shape=(MAX_NUM_CARS+1, MAX_NUM_CARS+1))

    # build the dynamics table
    transition_matrix = build_dynamics_table()
    print(transition_matrix)

    # now we will do policy iteration
    # we will use the following function to compute the value of a state given the current policy
    def policy_evaluation():
        # we will use the following function to compute the value of a state given the current policy
        def compute_value_function(state_x, state_y):
            # the value of a state is the expected reward of taking the current policy
            # plus the discounted value of the next state
            action = policy[state_x, state_y]
            next_state_x = min(max(0, state_x - action), 20)
            next_state_y = min(max(0, state_y + action), 20)

            # compute the expected reward
            expected_reward = transition_matrix[state_x, state_y, next_state_x, next_state_y, 1]

            # compute the discounted value of the next state
            discounted_value = DISCOUNT_FACTOR * value_function[next_state_x, next_state_y]

            return expected_reward + discounted_value

        # now we will iterate over all states and compute the value function
        for state_x in range(MAX_NUM_CARS+1):
            for state_y in range(MAX_NUM_CARS+1):
                value_function[state_x, state_y] = compute_value_function(state_x, state_y)

    # we will use the following function to compute the policy given the current value function
    # def policy_improvement():
    #     # we will use the following function to compute the value of a state given the current policy
    #     def compute_action_value(state_x, state_y, action):
    #         # the value of a state is the expected reward of taking the current policy
    #         # plus the discounted value of the next state
    #         next_state_x = min(max(0, state_x - action), 20)
    #         next_state_y = min(max(0, state_y + action), 20)

    #         # compute the expected reward
    #         expected_reward = transition_matrix[state_x, state_y, next_state_x, next_state_y, 1]

    #         # compute the discounted value of the next state
    #         discounted_value = DISCOUNT_FACTOR * value_function[next_state_x, next_state_y]

    #         return expected_reward + discounted_value

        # now we will iterate over all states and compute the value function
        #for state_x in range(MAX_NUM_CARS+



