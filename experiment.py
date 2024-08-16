import gym

env = gym.make("FrozenLake-v1", render_mode="rgb_array")
print(env.reset())