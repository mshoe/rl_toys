from envs.PuckWorldMagnet import PuckWorldMagnetEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
##  Train model
    env = make_vec_env(PuckWorldMagnetEnv, n_envs=4)
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_puckworld_magnet/")
    model.learn(total_timesteps=500000)
    model.save("ppo_PuckWorldMagnet")
    # todo: decide on nn 128 x 2 hidden layers, check if network output is clamped
    #todo: save during training and tensorboard
