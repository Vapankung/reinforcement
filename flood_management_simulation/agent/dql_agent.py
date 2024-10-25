# agent/dql_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .replay_memory import ReplayMemory, Transition

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 256]):
        super(DQN, self).__init__()
        layers = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DQLAgent:
    def __init__(self, state_size, device, actions_list):
        self.device = device
        self.state_size = state_size
        self.actions_list = actions_list
        self.num_actions = len(actions_list)

        self.online_net = DQN(state_size, self.num_actions).to(self.device)
        self.target_net = DQN(state_size, self.num_actions).to(self.device)
        self.update_target_network()

        self.memory = ReplayMemory(10000)
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 5000
        self.steps_done = 0

    def select_action(self, state):
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if np.random.rand() < epsilon:
            action_index = np.random.randint(0, self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                q_values = self.online_net(state_tensor)
                action_index = q_values.argmax().item()
        return action_index

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # Compute Q(s_t, a_t)
        current_q_values = self.online_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save_model(self, filepath):
        torch.save(self.online_net.state_dict(), filepath)

    def load_model(self, filepath):
        self.online_net.load_state_dict(torch.load(filepath))
        self.update_target_network()
