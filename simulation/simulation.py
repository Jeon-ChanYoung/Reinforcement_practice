import numpy as np
from PIL import Image
import imageio
import cv2

EPISODES = 40000
SIZE = 30
GAP = 2000

EPSILON_DECAY = 0.9999
LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
TARGET_N = 2
MOVE_PENALTY = 1
TARGET_REWARD = 30
color = {1 : (0, 0, 255),
         2 : (0, 255, 0)}

epsilon = 0.95

Q_table = {}
for x1 in range(-SIZE+1, SIZE):
    for y1 in range(-SIZE+1, SIZE):
        Q_table[((x1,y1))] = [np.random.random() for _ in range(4)]

class Game:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    
    def action(self, choice):
        if choice == 0:
            self.move(0, 1)
        elif choice == 1:
            self.move(0, -1)
        elif choice == 2:
            self.move(1, 0)
        else:
            self.move(-1, 0)
        
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)

    def move(self, x, y):
        self.x += x
        self.y += y

        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        
        if self.x >= SIZE:
            self.x = SIZE -1
        if self.y >= SIZE:
            self.y = SIZE -1

episode_rewards = [] 
frames = []

for episode in range(EPISODES+1):
    player = Game()
    target = Game()

    if episode % GAP == 0:
        show = True
        print(f"에피소드 : {episode} | 평균 : {np.mean(episode_rewards):.3f} | 앱실론 : {epsilon*100:3f}%")
    else:
        show = False
    
    episode_reward = 0
    for _ in range(100):
        obv = player - target
        if np.random.random() > epsilon:
            action = np.argmax(Q_table[obv])
        else:
            action = np.random.randint(0, 4)
        
        player.action(action)
        if player.x == target.x and player.y == target.y:
            reward = TARGET_REWARD
        else:
            reward = -MOVE_PENALTY
        
        new_obv = player - target
        current_q = Q_table[new_obv][action]
        maxQ = np.max(Q_table[new_obv])

        if reward == TARGET_REWARD:
            new_q = TARGET_REWARD
        else:
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * maxQ - current_q)
        
        Q_table[obv][action] = new_q
        episode_reward += reward

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[player.x][player.y] = color[PLAYER_N]
            env[target.x][target.y] = color[TARGET_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((500, 500), resample=Image.BOX)
            frames.append(img)
            cv2.imshow("", img)
            if reward == TARGET_REWARD:
                cv2.waitKey(1000)
                break
            else:
                cv2.waitKey(30)
                
        if reward == TARGET_REWARD:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPSILON_DECAY

imageio.mimsave('simulation.gif', frames, fps=20)