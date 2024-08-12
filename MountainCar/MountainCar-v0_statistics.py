import gym
import numpy as np
import warnings
import matplotlib.pyplot as plt
import cv2
import imageio

warnings.filterwarnings("ignore", category=DeprecationWarning)

env = gym.make("MountainCar-v0", render_mode="rgb_array")
env.reset()

a = 0.1  # 학습률
r = 0.95  # 감가율 (할인률)
EPISODES = 50000  # 총 에피소드 갯수
GAP = 500

epsilon = 0.5  # 앱실론e, 새로운 행동 시도 확률
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {"ep": [], "avg": [], "min": [], "max": []}

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

# 그래프 초기 설정
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(0, EPISODES)
ax.set_ylim(-200, 0)
line_avg, = ax.plot([], [], label="avg_reward")
line_min, = ax.plot([], [], label="min_reward")
line_max, = ax.plot([], [], label="max_reward")
ax.set_xlabel("Episodes")
ax.set_ylabel("Rewards")
plt.legend(loc=7)

# GIF를 위한 프레임 저장 리스트
frames = []

# 실시간 그래프 업데이트 함수
def update_plot():
    line_avg.set_data(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"])
    line_min.set_data(aggr_ep_rewards["ep"], aggr_ep_rewards["min"])
    line_max.set_data(aggr_ep_rewards["ep"], aggr_ep_rewards["max"])
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    
    # 캡처된 그래프를 프레임으로 저장
    fig.canvas.draw()
    # 그래프의 렌더링된 이미지 캡처
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape((600, 800, 3))
    
    # OpenCV의 BGR 색상 포맷으로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    frames.append(image)
    
    plt.pause(0.001)

# 학습 진행
for episode in range(EPISODES):
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset()[0])
    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(observation)
        done = terminated or truncated

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = current_q + a * (reward + r * max_future_q - current_q)
            q_table[discrete_state + (action, )] = new_q
        elif observation[0] >= env.goal_position:
            print(f"목표 도달 에피소드 : {episode}")
            q_table[discrete_state + (action, )] = 0
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)
    if episode % GAP == 0:
        average_reward = sum(ep_rewards[-GAP:]) / len(ep_rewards)
        MIN = min(ep_rewards[-GAP:])
        MAX = max(ep_rewards[-GAP:])
        aggr_ep_rewards["ep"].append(episode)
        aggr_ep_rewards["avg"].append(average_reward)
        aggr_ep_rewards["min"].append(MIN)
        aggr_ep_rewards["max"].append(MAX)
        
        update_plot()

env.close()

# GIF 생성
imageio.mimsave('mountaincar_training_stats.gif', frames, fps=20)

plt.ioff()
plt.show()
