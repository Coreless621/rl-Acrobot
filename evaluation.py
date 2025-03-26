import gymnasium as gym
import torch
from training import select_action, DQNetwork

env = gym.make("Acrobot-v1", render_mode = "human")
policy = DQNetwork(6,3)
policy.load_state_dict(torch.load("weights.pth"))
policy.eval()

for episode in range(3):
    state, _ = env.reset()
    done = False
    while not done:
        action = select_action(state, epsilon=0)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state

env.close()
