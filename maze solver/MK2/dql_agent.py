# dql_agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95          # Discount factor
        self.epsilon = 1.0         # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None

    def build_model(self, input_shape):
        # Convolutional Neural Network for variable input sizes
        class Net(nn.Module):
            def __init__(self, output_shape):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.fc1 = nn.Linear(64, 128)
                self.out = nn.Linear(128, output_shape)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.mean(x, dim=(2, 3))  # Global Average Pooling
                x = torch.relu(self.fc1(x))
                x = self.out(x)
                return x

        self.model = Net(self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        # Store experiences
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()

    def replay(self, batch_size):
        # Train the model using experiences
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        # Prepare batches efficiently
        states = np.stack([experience[0] for experience in minibatch])
        state_batch = torch.FloatTensor(states).unsqueeze(1).to(self.device)

        actions = np.array([experience[1] for experience in minibatch])
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)

        rewards = np.array([experience[2] for experience in minibatch])
        reward_batch = torch.FloatTensor(rewards).to(self.device)

        next_states = np.stack([experience[3] for experience in minibatch])
        next_state_batch = torch.FloatTensor(next_states).unsqueeze(1).to(self.device)

        dones = np.array([float(experience[4]) for experience in minibatch])
        done_batch = torch.FloatTensor(dones).to(self.device)

        # Compute current Q values
        q_values = self.model(state_batch).gather(1, action_batch).squeeze(1)

        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.model(next_state_batch).max(1)[0]
        # Compute target Q values
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path, input_shape):
        self.build_model(input_shape)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
