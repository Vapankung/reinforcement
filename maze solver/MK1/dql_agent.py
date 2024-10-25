import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95          # Discount factor
        self.epsilon = 1.0         # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        # Neural Net for Deep-Q learning
        class Net(nn.Module):
            def __init__(self, input_shape, output_shape):
                super(Net, self).__init__()
                self.flatten = nn.Flatten()
                n_input = np.prod(input_shape)
                self.fc1 = nn.Linear(n_input, 24)
                self.fc2 = nn.Linear(24, 24)
                self.out = nn.Linear(24, output_shape)

            def forward(self, x):
                x = self.flatten(x)
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.out(x)
                return x

        return Net(self.state_shape, self.action_size)

    def remember(self, state, action, reward, next_state, done):
        # Store experiences
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values[0]).item()

    def replay(self, batch_size):
        # Train the model using experiences
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        # Prepare batches
        state_batch = torch.FloatTensor(np.array([experience[0] for experience in minibatch])).to(self.device)
        action_batch = torch.LongTensor([experience[1] for experience in minibatch]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor([experience[2] for experience in minibatch]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([experience[3] for experience in minibatch])).to(self.device)
        done_batch = torch.FloatTensor([float(experience[4]) for experience in minibatch]).to(self.device)

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
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
