from envs.PuckWorld import PuckWorldEnv


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
##  Play as human
    # env = RBGravityControl("human", True)
    # env.reset()

    # while True:
    #     obs, rewards, done, info = env.step(env.action_space.sample())
    #     print(rewards)
    #     env.render()
    #     if done:
    #         obs = env.reset()

##  Train model
    env = make_vec_env(PuckWorldEnv, n_envs=4)
    env = PuckWorldEnv()
    model = PPO("MultiInputPolicy", env, verbose=1)
    # todo: decide on nn 128 x 2 hidden layers, check if network output is clamped
    model.learn(total_timesteps=1000000)
    model.save("ppo_PuckWorld")
    #todo: save during training and tensorboard

##  Run trained model
    # env = PuckWorldEnv(render_mode="human")
    # model = PPO.load("ppo_PuckWorld")

    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
        
    #     obs, rewards, done, info = env.step(action)
    #     env.render()
    #     if done:
    #         obs = env.reset()