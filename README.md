# 🤸 Acrobot-v1 – Deep Q-Learning Agent

This project implements a **Deep Q-Network (DQN)** to solve the `Acrobot-v1` environment from [Gymnasium](https://gymnasium.farama.org/).  
The agent learns to swing a two-link pendulum (Acrobot) upward into a balanced position using reinforcement learning and function approximation.

---

## ⚙️ Environment Overview

- **Environment:** `Acrobot-v1`
- **State space:** 6 continuous variables (angles and angular velocities)
- **Action space:** 3 discrete actions (`-1`, `0`, `+1` torque)
- **Goal:** Raise the tip of the lower link above a certain height
- **Reward structure (customized):**
  - `0` at every step (default is `-1`), I found the original one too extreme
  - Large positive reward (`10000 / step_count`) upon successful completion

---

## 🧠 Algorithm

- **Type:** Deep Q-Network (DQN)
- **Techniques used:**
  - Experience Replay Buffer
  - Target Network Updates
  - Epsilon-Greedy Exploration with Decay
  - Custom Reward Shaping for faster learning

---

## 🛠️ Hyperparameters

| Parameter        | Value       |
|------------------|-------------|
| Episodes         | `1000`      |
| Learning Rate    | `0.0001`    |
| Gamma            | `0.95`      |
| Epsilon (start)  | `0.95`      |
| Min Epsilon      | `0.1`       |
| Epsilon Decay    | `0.99`      |
| Batch Size       | `32`        |
| Replay Buffer    | `1,000,000` |

---

## 📁 Project Structure

| File             | Description |
|------------------|-------------|
| `training.py`    | Trains the DQN agent using experience replay and target network; saves `weights.pth` |
| `evaluation.py`  | Loads trained weights and runs the agent for 3 episodes with rendering |
| `weights.pth`    | (Generated) Trained model weights |

---

## 📊 Training Feedback

The script plots total reward per episode to track learning progress.  
Reward shaping speeds up convergence by giving the agent higher reward for quicker solutions.
