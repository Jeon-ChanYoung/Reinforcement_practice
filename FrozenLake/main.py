import gym
import numpy as np
import torch as T
from DQN_model import Agent 
import cv2
import imageio

def modified_step(env, action):
    state, reward, done, trun, info = env.step(action)

    if done and reward == 1:
        reward = 10
    elif done and reward == 0:
        reward = -1
    return state, reward, done, trun, info

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
n_games = 1000
GAP = 50  # visualize gap
agent = Agent(gamma=0.99, 
              epsilon=1.0, 
              lr=0.001, 
              input_dims=[1], 
              batch_size=64, 
              n_actions=4,
              eps_dec=0.0002)

scores = []
eps_history = []
frames = []

for episode in range(1, n_games+1):
    score = 0
    done = False
    observation = env.reset()[0]

    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, trunc, info = modified_step(env, action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_

        # 매 GAP번째 에피소드마다 시각화
        if episode % GAP == 0:
            frame = env.render()  # 'rgb_array' 모드로 초기화했으므로 화면을 얻습니다.
            frames.append(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('FrozenLake', frame)  # 시각화
            cv2.waitKey(60)

    scores.append(score)
    eps_history.append(agent.epsilon)

    avg_score = np.mean(scores[-10:])
    print(f'| 에피소드: {episode} | 평균 점수: {avg_score:.2f} | 앱실론: {agent.epsilon:.2f} |')

env.close()
cv2.destroyAllWindows()
imageio.mimsave("FrozenLake-v1.gif", frames, fps=20)