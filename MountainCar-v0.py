import gym
import cv2
import numpy as np

env = gym.make("MountainCar-v0", render_mode="rgb_array")  #render_mode="human"
env.reset()
a = 0.1 # 학습률
r = 0.95 # 감가율 (할인률)
EPISODES = 2500 # 총 에피소그 갯수 
GAP = 500

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
'''
print(env.observation_space) # Box([-1.2  -0.07], [0.6  0.07], (2,), float32)
print(env.action_space)      #      low             high
print(discrete_os_win_size)
print(q_table.shape)
'''

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

for episode in range(1, EPISODES):
    discrete_state = get_discrete_state(env.reset()[0])
    done = False

    while not done:
        if episode % GAP == 0:
            img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("test", img)
            cv2.waitKey(50)

        action = np.argmax(q_table[discrete_state])
        observation, reward, terminated, truncated, info = env.step(action)
        new_discrete_state = get_discrete_state(observation)
        done = terminated or truncated # 정상에 도달 여부 or 제한시간 지나면 False
        
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = current_q + a * (reward + r * max_future_q - current_q)
            q_table[discrete_state + (action, )] = new_q
            # Q(s,a) ← Q(s,a)+α×(r+γ×maxQ(s′,a)−Q(s,a))
            
        elif observation[0] >= env.goal_position:
            print(f"목표 도달 에피소드 : {episode}")
            q_table[discrete_state + (action, )] = 0
        discrete_state = new_discrete_state

env.close()
