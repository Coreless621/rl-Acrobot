# DQN for Acrobot env from gymnasium

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt

import gymnasium as gym

# parameters
num_episodes = 1_000
state_size = 6
action_size = 3
learning_rate = 0.0001
gamma = 0.95
epsilon = 0.95
min_epsilon = 0.1
epsilon_decay = 0.99



memory_size = 1_000_000
batch_size = 32
episode_rewards = []

env = gym.make("Acrobot-v1")

replay_buffer = deque(maxlen=memory_size)

# Defining neural network
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x) # outputting q values for all actions

policy_net = DQNetwork(state_size, action_size)
target_net = DQNetwork(state_size, action_size) # target network
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

def select_action(state, epsilon):
    if np.random.uniform() < epsilon:
        action = env.action_space.sample()
        return action
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state)
        return q_values.argmax(dim=1).item()

def store_transition(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

def train():
    if len(replay_buffer) < batch_size:
        return
    
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.FloatTensor(dones).unsqueeze(1)

    # computing q values
    q_values = policy_net(states).gather(1, actions)

    # computing max q_s'a' for next states using target network
    next_q_values = target_net(next_states).max(1, keepdim=True)[0]
    next_q_values[dones == 1] = 0.0


    #computing target q_values
    target_q_values = rewards + gamma * next_q_values * (1-dones)

    # computing loss
    loss = loss_fn(q_values, target_q_values)

    # optimizing model 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# loading state dicts if available
try:
    policy_net.load_state_dict(torch.load("weights.pth"))
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model not found. Training from scratch.")

# copying state dict to target net 
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# training loop
if __name__ == "__main__":
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        total_reward = 0
        step = 0
        done = False
        while not done:
            step+=1
            action = select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done is True:
                reward = 10000/step # modified reward system because original one is weak
            else:
                reward = 0

            store_transition(state, action, reward, next_state, done)
            train()
            state = next_state
            total_reward += reward
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict()) # update target network param every 10 episodes

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Avg Reward last 50: {avg_reward:.2f}, Epsilon: {epsilon:.3f}") # computing mean reward over last 50 episodes
        episode_rewards.append(total_reward)

    print(f"Completed training with avg last 100 episodes reward: {np.mean(episode_rewards[-100:])}")
    torch.save(policy_net.state_dict(), "weights.pth")


    plt.plot(episode_rewards, label="Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Performance over Time")
    plt.legend()
    plt.show()