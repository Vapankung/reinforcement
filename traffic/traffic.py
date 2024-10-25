import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import pygame
import os
import sys
import pickle  # Ensure this line is present

# ----------------------------
# CarAgent Class
# ----------------------------

class CarAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Dueling DQN networks
        self.policy_net = DuelingDQN(state_size, action_size)
        self.target_net = DuelingDQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = PrioritizedReplayMemory(10000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.gamma = 0.99
        self.batch_size = 64
        self.update_target_every = 10  # Episodes

        # Epsilon parameters for dynamic decay
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.steps_done = 0

    def select_action(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32)
                q_values = self.policy_net(state.unsqueeze(0))
                return torch.argmax(q_values).item()

    def memorize(self, state, action, reward, next_state, done, td_error):
        self.memory.push((state, action, reward, next_state, done), td_error)

    def learn(self, beta=0.4):
        if len(self.memory) < self.batch_size:
            return

        experiences, indices, weights = self.memory.sample(self.batch_size, beta)
        if len(experiences) == 0:
            return

        batch = list(zip(*experiences))

        # Convert list of numpy arrays to a single numpy array before tensor conversion
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32)
        actions = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Double DQN - action selection from policy net, value from target net
        next_actions = torch.argmax(self.policy_net(next_states), dim=1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute TD errors
        td_errors = expected_q_values.detach() - current_q_values
        loss = (weights * td_errors.pow(2)).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        td_errors = td_errors.detach().numpy()
        self.memory.update_priorities(indices, td_errors)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ----------------------------
# TrafficEnvironment Class
# ----------------------------

class TrafficEnvironment:
    def __init__(self, max_steps=1000, variable_patterns=False):
        self.max_steps = max_steps
        self.time_step = 0
        self.done = False
        self.light_phase = 0  # 0: NS green, 1: EW green
        self.current_phase_duration = 0
        self.max_phase_duration = 60
        self.traffic_pattern = 'normal'
        self.variable_patterns = variable_patterns
        self.cars = {'north': deque(), 'south': deque(), 'east': deque(), 'west': deque()}
        self.state_size = 6  # [north_queue, south_queue, east_queue, west_queue, light_phase, current_phase_duration]
        self.action_space = [0, 1]  # Actions: keep phase, switch phase
        self.collision_penalty = -50  # Penalty for collision

        # Initialize Pygame
        pygame.init()
        self.width = 600
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Traffic Simulation")
        self.clock = pygame.time.Clock()

        # Colors
        self.bg_color = (220, 220, 220)
        self.road_color = (50, 50, 50)
        self.car_color = (0, 0, 255)
        self.light_green = (0, 255, 0)
        self.light_red = (255, 0, 0)
        self.font = pygame.font.SysFont(None, 24)

        # Car settings
        self.car_size = 20
        self.MAX_SPEED = 5  # Maximum speed
        self.ACCELERATION = 0.2  # Acceleration rate
        self.DECELERATION = 0.5  # Deceleration rate
        self.SAFE_DISTANCE = 30  # Minimum safe distance between cars

        # Initialize CarAgents per direction
        self.car_agents = {
            'north': CarAgent(state_size=4, action_size=3),  # [speed, distance, light_state, phase_duration]
            'south': CarAgent(state_size=4, action_size=3),
            'east': CarAgent(state_size=4, action_size=3),
            'west': CarAgent(state_size=4, action_size=3)
        }

        # Action mapping for cars
        self.car_actions = {
            0: 'decelerate',
            1: 'maintain',
            2: 'accelerate'
        }

    def reset(self):
        self.time_step = 0
        self.done = False
        self.light_phase = 0
        self.current_phase_duration = 0
        self.cars = {'north': deque(), 'south': deque(), 'east': deque(), 'west': deque()}
        self._generate_traffic_pattern()
        return self._get_state()

    def _generate_traffic_pattern(self):
        if self.variable_patterns:
            patterns = ['normal', 'rush_hour', 'accident']
            self.traffic_pattern = random.choice(patterns)
        else:
            self.traffic_pattern = 'normal'

    def _car_arrival_rate(self, direction):
        if self.traffic_pattern == 'normal':
            return 0.2  # Reduced arrival rate
        elif self.traffic_pattern == 'rush_hour':
            if direction in ['north', 'south']:
                return 0.4
            else:
                return 0.1
        elif self.traffic_pattern == 'accident':
            if direction == 'east':
                return 0.05  # Less arrival due to accident
            else:
                return 0.2

    def step(self, action, episode):
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Apply action for traffic light agent
        reward_delay_penalty = -5
        if action == 1 and self.current_phase_duration >= 10:
            self.light_phase = 1 - self.light_phase  # Switch phase
            self.current_phase_duration = 0

        # Update current phase duration
        self.current_phase_duration += 1

        # Simulate car arrivals
        for direction in self.cars.keys():
            arrival_rate = self._car_arrival_rate(direction)
            if np.random.rand() < arrival_rate:
                # Each car is represented as a dictionary with position, arrival time, speed
                if direction == 'north':
                    initial_position = self.height - self.car_size - 10
                elif direction == 'south':
                    initial_position = 10
                elif direction == 'east':
                    initial_position = self.width - self.car_size - 10
                elif direction == 'west':
                    initial_position = 10
                # Add car with spacing to prevent immediate collisions
                if len(self.cars[direction]) == 0:
                    self.cars[direction].append({'position': initial_position, 'arrival_time': self.time_step, 'speed': 0})
                else:
                    last_car = self.cars[direction][-1]
                    if direction in ['north', 'west']:
                        distance = last_car['position'] - initial_position
                    else:
                        distance = initial_position - last_car['position']
                    if distance > self.SAFE_DISTANCE:
                        self.cars[direction].append({'position': initial_position, 'arrival_time': self.time_step, 'speed': 0})

        # Process car movements and actions
        self._update_car_positions()

        # Calculate rewards for traffic light agent
        waiting_times = []
        for queue in self.cars.values():
            waiting_times.extend([self.time_step - car['arrival_time'] for car in queue])

        total_waiting_time = sum(waiting_times)
        num_waiting_cars = len(waiting_times)
        reward = - (num_waiting_cars + 0.1 * total_waiting_time)

        # Penalty for overextending phase duration
        if self.current_phase_duration > self.max_phase_duration:
            reward += reward_delay_penalty

        # Collision detection and punishment
        collisions = self._detect_collisions()
        if collisions > 0:
            reward += self.collision_penalty * collisions  # Apply penalty for each collision

        # Update time step
        self.time_step += 1

        # Check if simulation is done
        if self.time_step >= self.max_steps:
            self.done = True

        # Render the environment with episode information
        self.render(episode)

        return self._get_state(), reward, self.done, {}

    def _detect_collisions(self):
        collisions = 0
        for direction, queue in self.cars.items():
            sorted_queue = sorted(queue, key=lambda x: x['position'], reverse=(direction in ['north', 'west']))
            for i in range(1, len(sorted_queue)):
                if direction in ['north', 'south']:
                    distance = sorted_queue[i-1]['position'] - sorted_queue[i]['position']
                else:
                    distance = sorted_queue[i]['position'] - sorted_queue[i-1]['position']
                if distance < self.car_size * 1.5:
                    collisions += 1
        return collisions

    def _get_state(self):
        # State includes number of cars waiting in each direction and current phase
        state = []
        for queue in self.cars.values():
            state.append(len(queue))
        state.extend([self.light_phase, self.current_phase_duration])
        return np.array(state, dtype=np.float32)

    def _update_car_positions(self):
        for direction, queue in self.cars.items():
            agent = self.car_agents[direction]
            cars_to_remove = []  # Collect cars to remove after iteration
            for idx, car in enumerate(queue):
                # Define state for the agent
                if idx == 0:
                    distance = float('inf')  # No car ahead
                else:
                    if direction in ['north', 'west']:
                        distance = car['position'] - queue[idx - 1]['position']
                    else:
                        distance = queue[idx - 1]['position'] - car['position']

                traffic_light_state = self.light_phase  # Assuming binary state

                state = np.array([
                    car['speed'],
                    distance if distance != float('inf') else 100,  # Assign a large distance if no car ahead
                    traffic_light_state,
                    self.current_phase_duration
                ], dtype=np.float32)

                # Select action
                action = agent.select_action(state)

                # Execute action
                if action == 0:  # Decelerate
                    car['speed'] = max(car['speed'] - self.DECELERATION, 0)
                elif action == 1:  # Maintain
                    pass  # No change in speed
                elif action == 2:  # Accelerate
                    car['speed'] = min(car['speed'] + self.ACCELERATION, self.MAX_SPEED)

                # Update position based on direction and current speed
                if direction == 'north':
                    car['position'] -= car['speed']
                elif direction == 'south':
                    car['position'] += car['speed']
                elif direction == 'east':
                    car['position'] -= car['speed']
                elif direction == 'west':
                    car['position'] += car['speed']

                # Calculate next state and reward
                if idx == 0:
                    next_distance = float('inf')
                else:
                    if direction in ['north', 'west']:
                        next_distance = car['position'] - queue[idx - 1]['position']
                    else:
                        next_distance = queue[idx - 1]['position'] - car['position']

                next_traffic_light_state = self.light_phase
                next_phase_duration = self.current_phase_duration

                next_state = np.array([
                    car['speed'],
                    next_distance if next_distance != float('inf') else 100,
                    next_traffic_light_state,
                    next_phase_duration
                ], dtype=np.float32)

                # Define reward
                if next_distance < self.SAFE_DISTANCE:
                    reward = -10  # Collision penalty
                else:
                    reward = -1  # Small penalty for time spent

                done = False  # Define termination condition if needed

                # Compute TD error
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                    q_value = agent.policy_net(state_tensor.unsqueeze(0))[0][action]
                    next_q_value = agent.target_net(next_state_tensor.unsqueeze(0)).max().item()
                    expected_q_value = reward + agent.gamma * next_q_value * (1 - done)
                    td_error = expected_q_value - q_value.item()

                # Memorize and learn
                agent.memorize(state, action, reward, next_state, done, td_error)
                agent.learn()

                # Mark cars that have left the screen for removal
                if direction in ['north', 'east'] and car['position'] < -self.car_size:
                    cars_to_remove.append(car)
                elif direction in ['south', 'west'] and car['position'] > self.width + self.car_size:
                    cars_to_remove.append(car)

            # After iteration, remove cars that have left the screen
            for car in cars_to_remove:
                queue.remove(car)

    def render(self, episode):
        # Clear screen
        self.screen.fill(self.bg_color)

        # Draw roads
        pygame.draw.rect(self.screen, self.road_color, (self.width // 2 - 100, 0, 200, self.height))
        pygame.draw.rect(self.screen, self.road_color, (0, self.height // 2 - 100, self.width, 200))

        # Draw traffic lights
        if self.light_phase == 0:  # NS green
            ns_color = self.light_green
            ew_color = self.light_red
        else:  # EW green
            ns_color = self.light_red
            ew_color = self.light_green

        # North-South lights
        pygame.draw.circle(self.screen, ns_color, (self.width // 2, self.height // 2 - 120), 15)
        pygame.draw.circle(self.screen, ns_color, (self.width // 2, self.height // 2 + 120), 15)
        # East-West lights
        pygame.draw.circle(self.screen, ew_color, (self.width // 2 - 120, self.height // 2), 15)
        pygame.draw.circle(self.screen, ew_color, (self.width // 2 + 120, self.height // 2), 15)

        # Draw cars with speed-based color
        for direction, queue in self.cars.items():
            for idx, car in enumerate(queue):
                speed_ratio = car['speed'] / self.MAX_SPEED
                # Green to red color based on speed
                car_color = (
                    int(255 * (1 - speed_ratio)),
                    int(255 * speed_ratio),
                    0
                )
                if direction == 'north':
                    x = self.width // 2 - 50
                    y = car['position']
                elif direction == 'south':
                    x = self.width // 2 + 30
                    y = car['position']
                elif direction == 'east':
                    x = car['position']
                    y = self.height // 2 + 30
                elif direction == 'west':
                    x = car['position']
                    y = self.height // 2 - 50
                pygame.draw.rect(self.screen, car_color, (x, y, self.car_size, self.car_size))

        # Draw countdown timer and episode
        timer_text = self.font.render(f"Phase Time Left: {self.max_phase_duration - self.current_phase_duration}", True, (0, 0, 0))
        self.screen.blit(timer_text, (10, 10))
        episode_text = self.font.render(f"Episode: {episode}", True, (0, 0, 0))
        self.screen.blit(episode_text, (10, 40))

        # Update display
        pygame.display.flip()

        # Control simulation speed
        self.clock.tick(60)  # 60 FPS

    def close(self):
        pygame.quit()

# ----------------------------
# Dueling Double DQN Network
# ----------------------------

class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU()

        # Value stream
        self.fc_value = nn.Linear(128, 64)
        self.value = nn.Linear(64, 1)

        # Advantage stream
        self.fc_advantage = nn.Linear(128, 64)
        self.advantage = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))

        val = self.relu(self.fc_value(x))
        val = self.value(val)

        adv = self.relu(self.fc_advantage(x))
        adv = self.advantage(adv)

        # Combine value and advantage streams
        q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_vals

# ----------------------------
# Prioritized Experience Replay Memory
# ----------------------------

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def push(self, experience, td_error):
        priority = (abs(td_error) + 1e-5) ** self.alpha
        self.memory.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == 0:
            return [], [], []
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sample_probs = scaled_priorities / np.sum(scaled_priorities)

        indices = np.random.choice(len(self.memory), batch_size, p=sample_probs)
        experiences = [self.memory[i] for i in indices]
        weights = (len(self.memory) * sample_probs[indices]) ** (-beta)
        weights /= weights.max()
        return experiences, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-5)

    def __len__(self):
        return len(self.memory)

# ----------------------------
# Agent Class for Traffic Lights
# ----------------------------

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Dueling DQN networks
        self.policy_net = DuelingDQN(state_size, action_size)
        self.target_net = DuelingDQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = PrioritizedReplayMemory(10000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.gamma = 0.99
        self.batch_size = 64
        self.update_target_every = 10  # Episodes

        # Epsilon parameters for dynamic decay
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.steps_done = 0

    def select_action(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32)
                q_values = self.policy_net(state.unsqueeze(0))
                return torch.argmax(q_values).item()

    def memorize(self, state, action, reward, next_state, done, td_error):
        self.memory.push((state, action, reward, next_state, done), td_error)

    def learn(self, beta=0.4):
        if len(self.memory) < self.batch_size:
            return

        experiences, indices, weights = self.memory.sample(self.batch_size, beta)
        if len(experiences) == 0:
            return

        batch = list(zip(*experiences))

        # Convert list of numpy arrays to a single numpy array before tensor conversion
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32)
        actions = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Double DQN - action selection from policy net, value from target net
        next_actions = torch.argmax(self.policy_net(next_states), dim=1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute TD errors
        td_errors = expected_q_values.detach() - current_q_values
        loss = (weights * td_errors.pow(2)).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        td_errors = td_errors.detach().numpy()
        self.memory.update_priorities(indices, td_errors)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ----------------------------
# Training Loop with Enhancements
# ----------------------------

def train_agent(traffic_agent, car_agents, env, episodes, early_stopping_threshold=0.01):
    total_rewards = []
    moving_avg_rewards = []
    best_avg_reward = float('-inf')
    patience = 10  # Early stopping patience
    patience_counter = 0

    # Initialize Matplotlib figure for real-time plotting
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title('Traffic Light Agent Performance Over Episodes')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    line, = ax.plot([], [], label='Total Reward')
    ax.legend()
    plt.show()

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select and execute action for traffic light agent
            action = traffic_agent.select_action(state)
            next_state, reward, done, _ = env.step(action, episode)
            total_reward += reward

            # Compute TD error for traffic light agent
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                q_value = traffic_agent.policy_net(state_tensor.unsqueeze(0))[0][action]
                next_q_value = traffic_agent.target_net(next_state_tensor.unsqueeze(0)).max().item()
                expected_q_value = reward + traffic_agent.gamma * next_q_value * (1 - done)
                td_error = expected_q_value - q_value.item()

            # Memorize and learn for traffic light agent
            traffic_agent.memorize(state, action, reward, next_state, done, td_error)
            traffic_agent.learn()

            state = next_state

        # Update target network for traffic light agent
        if (episode + 1) % traffic_agent.update_target_every == 0:
            traffic_agent.update_target_network()

        total_rewards.append(total_reward)

        # Update Matplotlib plot
        line.set_xdata(range(1, len(total_rewards) + 1))
        line.set_ydata(total_rewards)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

        # Moving average for early stopping
        if len(total_rewards) >= 20:
            moving_avg = np.mean(total_rewards[-20:])
            moving_avg_rewards.append(moving_avg)

            # Early stopping check
            if moving_avg > best_avg_reward + early_stopping_threshold:
                best_avg_reward = moving_avg
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at episode {episode + 1}")
                break

        # Save progress every 50 episodes
        if (episode + 1) % 50 == 0:
            save_progress(traffic_agent, total_rewards, episode + 1)
            print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {traffic_agent.epsilon:.2f}")

    env.close()
    plt.ioff()
    plt.show()
    return total_rewards

# ----------------------------
# Saving and Loading Progress
# ----------------------------

def save_progress(agent, total_rewards, episode):
    checkpoint = {
        'policy_net_state_dict': agent.policy_net.state_dict(),
        'target_net_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'memory': agent.memory,
        'total_rewards': total_rewards,
        'episode': episode,
        'epsilon': agent.epsilon
    }
    with open('dqn_checkpoint.pkl', 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Progress saved at episode {episode}.")

def load_progress(agent):
    with open('dqn_checkpoint.pkl', 'rb') as f:
        checkpoint = pickle.load(f)
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.memory = checkpoint['memory']
    total_rewards = checkpoint['total_rewards']
    episode = checkpoint['episode']
    agent.epsilon = checkpoint['epsilon']
    print(f"Progress loaded from episode {episode}.")
    return total_rewards, episode

# ----------------------------
# Main Execution
# ----------------------------

def main():
    env = TrafficEnvironment(variable_patterns=True)

    state_size = env.state_size
    action_size = len(env.action_space)

    # Initialize traffic light agent
    traffic_agent = Agent(state_size, action_size)

    # Initialize car agents per direction
    car_agents = env.car_agents  # Already initialized in TrafficEnvironment

    episodes = 1000

    # Check if previous progress exists
    if os.path.exists('dqn_checkpoint.pkl'):
        total_rewards, start_episode = load_progress(traffic_agent)
    else:
        total_rewards = []
        start_episode = 0

    try:
        rewards = train_agent(traffic_agent, car_agents, env, episodes - start_episode)
        total_rewards.extend(rewards)
    except KeyboardInterrupt:
        save_progress(traffic_agent, total_rewards, start_episode)
        print("Training interrupted and progress saved.")

    # Plot final performance
    plot_performance(total_rewards)

def plot_performance(total_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(total_rewards) + 1), total_rewards, label='Total Rewards')
    plt.title('Traffic Light Agent Performance Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
