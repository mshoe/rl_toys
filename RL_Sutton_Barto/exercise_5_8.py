from racetrack import Racetrack
import numpy as np
from MonteCarloOnPolicyES import MontoCarloOnPolicyES



if __name__ == "__main__":
    env = Racetrack(max_speed=2, render_mode="none")

    # total_obs_space = np.array([], dtype=np.int)
    # for key in env.observation_space:
    #     print(key, env.observation_space[key].low)
    #     print(key, env.observation_space[key].high)

    #     print(env.observation_space[key].high - env.observation_space[key].low + 1)
    #     total_obs_space = np.append(total_obs_space, env.observation_space[key].high - env.observation_space[key].low + 1)

    # print(total_obs_space)
    # print("action space shape", env.action_space.n)
    # # try on policy monto carlo control

    # print("start cells =", env.start_cells)

    #test policy
    MCpolicy = MontoCarloOnPolicyES(env)
    print(MCpolicy.policy.shape)
    print(MCpolicy.policy[tuple(np.array([0,1,2,2]))])
    print(MCpolicy.Q[tuple(np.array([0,1,2,2]))])
    MCpolicy.learn(0.9, 1000, 100,verbose=False)

 

    ## TODO: print the policy and Q function, and plot the value function
    ## TODO: implement offpolicy monte carlo prediction and control

    value_fn = MCpolicy.get_value_function()
    print("value function")
    print(value_fn)

    # while True:
    #     #action, _states = model.predict(obs)
    #     action = env.action_space.sample()
    #     obs, rewards, done, info = env.step(action)
    #     print("action =",env._action_to_acceleration[action])
    #     print("obs =",obs)
    #     print("info =",info)

    #     env.render()
    #     if done:
    #        obs, info = env.reset()
    