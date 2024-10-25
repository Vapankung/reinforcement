import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MAX_EPISODES = 500
MAX_STEPS = 1600
ENV_NAME = "BipedalWalker-v3"

# Neural Network for DDQL
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Using tanh to limit output between -1 and 1 for continuous actions

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

# Continuous action selection
def select_continuous_action(state, network):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_values = network(state_tensor)
        return action_values.squeeze(0).numpy()

# Training function
def train_ddql():
    env = gym.make(ENV_NAME, render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize networks and replay buffer
    online_net = DQNetwork(state_dim, action_dim)
    target_net = DQNetwork(state_dim, action_dim)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    epsilon = EPSILON_START
    global_step = 0
    rewards_per_episode = []

    # Setup for plotting
    plt.ion()  # Turn interactive mode on
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    line, = ax.plot([], [], 'r-')

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False

        for step in range(MAX_STEPS):
            global_step += 1

            # No need to call env.render() as rendering is handled by render_mode='human'

            # Select continuous action from the neural network
            action = select_continuous_action(state, online_net)

            # Execute action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Train the network if we have enough experience in the replay buffer
            if replay_buffer.size() > MIN_REPLAY_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                # Convert to tensors
                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # Online network predicts the Q-values for actions
                q_values = online_net(states)

                # Target network computes the target Q-values
                with torch.no_grad():
                    next_q_values_online = online_net(next_states)
                    best_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                    next_q_values_target = target_net(next_states).gather(1, best_actions)
                    target_q_values = rewards + GAMMA * next_q_values_target * (1 - dones)

                # Compute loss (Mean Squared Error)
                loss = nn.MSELoss()(q_values, target_q_values)

                # Backpropagate loss and update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the target network at regular intervals
                if global_step % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(online_net.state_dict())

            # Epsilon decay for exploration-exploitation trade-off
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            if done:
                break

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Epsilon: {epsilon}")

        # Update plot after each episode
        line.set_xdata(np.arange(len(rewards_per_episode)))
        line.set_ydata(rewards_per_episode)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    env.close()

    # After training, plot the final rewards graph
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train_ddql()
