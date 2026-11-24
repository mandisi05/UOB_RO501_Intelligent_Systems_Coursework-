
#import called libraries 
import gymnasium as gym
import numpy as np
from collections import deque 
import random
import copy 

#Determine gymnasium simulation environment type, size and render mode and state
env = gym.make("FrozenLake-v1", map = "4x4", render_mode = "ansi", is_slippery = False)

#Group network attributes
class SimpleQNetworks:
    def __init__ (self,state_size, action_size,lr):
        self.lr = lr
        self.W1 = np.random.randn(state_size,64) *np.sqrt(2.0/state_size)
        self.W2 = np.random.randn(64,action_size) *sqrt(2.0/64) 

    def forward (self,x):
        self.x = x
        #hidden relu 
        self.h = np.maximum(0, np.dot(x, self.W1))
        self.q = np.dot(self.h, self.W2)
        return self.q

    def backward(self, action, target):
        error = self.q[0, action] - target
        dW2 = np.zeros_like(self.W2)
        dW2[:, action] = error * self.h[0]
        # Backpropagation through the relu derivative
        dh = error * self.W2[:, action]
        dh[self.h[0] <= 0] = 0
        dW1 = np.outer(self.x[0], dh)
        self.W2 -= self.lr * np.clip(dW2, -1, 1)
        self.W1 -= self.lr * np.clip(dW1, -1, 1)

class FrozenLakeAgent:
    def __init__(self, env, learning_rate):
        self.env = env
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.q_net = SimpleQNetwork(self.state_size, self.action_size, learning_rate)
        self.target_net = copy.deepcopy(self.q_net)
        self.epsilon = 1.0
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.steps_done = 0

    def state_to_vector(self, state):
        v = np.zeros((1, self.state_size))
        v[0, state] = 1.0
        return v

    def get_action(self, obs, eval_mode=False):# eval_mode (testing),pick the best
        if not eval_mode and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        q_values = self.q_net.forward(self.state_to_vector(obs))
        return np.argmax(q_values[0])

    def learn(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        for obs, action, reward, terminated, next_obs in batch:
            self.q_net.forward(self.state_to_vector(obs))
            target = reward
            if not terminated:
                next_q = self.target_net.forward(self.state_to_vector(next_obs))
                target += 0.99 * np.max(next_q[0])
                self.q_net.backward(action, target)
            self.steps_done += 1
        if self.steps_done % 100 == 0:
            self.target_net.W1 = self.q_net.W1.copy()
            self.target_net.W2 = self.q_net.W2.copy()
# Training
agent = FrozenLakeAgent(env, learning_rate=0.005)
epochs = 2000
verbose = True
success_history = deque(maxlen=100)
print("Starting Training")

for epoch in range(epochs):
    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
            # Agent Reward
        if terminated and reward == 1.0:
            reward = 5.0
            success_history.append(1)
        elif terminated and reward == 0.0:
            reward = -2.0
            success_history.append(0)
        else:
            reward = -0.01
    agent.memory.append((obs, action, reward, terminated, next_obs))
    agent.learn()
    obs = next_obs
    agent.epsilon = max(0.01, agent.epsilon - 0.001)
if verbose and epoch % 200 == 0:
    win_rate = (sum(success_history)/len(success_history))*100 if success_history else 0
    print(f"Epoch {epoch} | Win Rate: {win_rate}%")
env.close()
# Testing
print("\n render in human mode")
test_env = gym.make("FrozenLake-v1", map_name="4x4", render_mode="human", is_slippery=False)
for i in range(5):
    obs, _ = test_env.reset()
    done = False
    while not done:
        action = agent.get_action(obs, eval_mode=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
test_env.close()
