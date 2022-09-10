from envs.PuckWorldMagnet import PuckWorldMagnetEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
##  Play as human
    # env = PuckWorldMagnetEnv("human", True)
    # env.reset()

    # while True:
    #     obs, rewards, done, info = env.step(env.action_space.sample())
    #     #print(rewards)
    #     env.render()
    #     if done:
    #         obs = env.reset()

##  Run trained model
    env = PuckWorldMagnetEnv(render_mode="human")
    model = PPO.load("ppo_PuckWorldMagnet")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()