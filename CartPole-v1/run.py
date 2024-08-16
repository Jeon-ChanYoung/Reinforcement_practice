import gym
from FrozenLake.DQN_model import Agent
import numpy as np
import cv2
import imageio

env = gym.make("CartPole-v1", render_mode="rgb_array")
n_games = 120
GAP = 15  # visualize gap
agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=[4], batch_size=64, n_actions=2)
"""
input_dimes=[4] :
1. 카트위치
2. 카트속도
3. 막대기 각도
4. 막대기 각속도
"""
scores = []
eps_history = []
frames = []

for episode in range(1, n_games+1):
    score = 0
    done = False
    observation = env.reset()[0]

    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, trun, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()
        observation = observation_

        if episode % GAP == 0: 
            frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)  # 막대 기울기가 너무 기울어지면 시각화 종료
            frames.append(frame)
            cv2.imshow('CartPole-v1', frame)  # 시각화
            cv2.waitKey(1)

    scores.append(score)
    eps_history.append(agent.epsilon)

    avg_score = np.mean(scores[-10:])
    print(f'에피소드 : {episode} | 평균 : {avg_score:.2f} | 앱실론 : {agent.epsilon:.2f}')

env.close()
cv2.destroyAllWindows()  

imageio.mimsave("cartpole-v1.gif", frames, fps=30)
