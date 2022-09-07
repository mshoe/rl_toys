import gym
from stable_baselines3 import DQN
from envs.grid_world import GridWorldEnv

grid_size = 4
env = GridWorldEnv(size=grid_size)

model = DQN("MultiInputPolicy", env, verbose=1, buffer_size=10000, learning_starts=1000, learning_rate=0.099)
model.learn(total_timesteps=100000, log_interval=1)
model.save("dqn_grid_world"+str(grid_size))
#del model # remove to demonstrate saving and loading
#model = DQN.load("dqn_grid_world10")

env = GridWorldEnv(size=grid_size, render_mode="human")
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(action, reward)
    env.render()
    if done:
      obs = env.reset()