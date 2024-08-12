import gym
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
"""
DeprecationWarning: np.bool8 is a deprecated alias 
for np.bool_.  (Deprecated NumPy 1.24)
"""
env = gym.make("MountainCar-v0", render_mode="rgb_array")  #render_mode="human"
env.reset()
a = 0.1 # 학습률
r = 0.95 # 감가율 (할인률)
EPISODES = 5000 # 총 에피소그 갯수 
GAP = 500

epsilon = 0.5 # 앱실론e, 새로운 행동 시도 확률
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

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

for episode in range(EPISODES+1):
    discrete_state = get_discrete_state(env.reset()[0])
    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        if episode % GAP == 0:
            img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("test", img)
            cv2.waitKey(50)

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
    
    # 범위내 앱실론 선형적으로 감소
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()
