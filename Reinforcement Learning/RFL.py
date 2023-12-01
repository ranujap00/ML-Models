# %%
# !pip install stable-baselines3[extra]

# %%
import os
import gym
from stable_baselines3 import PPO # RL algorithm
from stable_baselines3.common.vec_env import DummyVecEnv # train agents on multiple envs
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
from IPython import display

# %%
# openAI gym allows you to build simulated envs easily
env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="rgb_array")

# %%
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        # env.render() # view the graphical representation of the env

        plt.imshow(env.render())
        display.display(plt.gcf())    
        display.clear_output(wait=True)

        action = env.action_space.sample()
        n_state, reward, done, truncated, info = env.step(action) # apply the action to the env
        score += reward
    
    print(f"Episode:{episode} Score: {score}")
# env.close()

# %%
env.action_space.sample()


