# main.py

import numpy as np
import torch
from agent import DQLAgent, generate_discrete_actions
from environment import Environment
from visualization import Visualization
import matplotlib.pyplot as plt
import os
import time  # To implement delays for visualization

def compute_reward(zones, pumps, alpha=1.0, beta=0.1):
    """
    Compute the reward based on flooding and energy consumption.

    Parameters:
    - zones: Dictionary of Zone objects.
    - pumps: Dictionary of Pump objects.
    - alpha: Weight for flooding.
    - beta: Weight for energy consumption.

    Returns:
    - reward: Computed reward value.
    """
    total_flood_volume = sum(max(zone.current_water_level - zone.flood_threshold, 0.0) for zone in zones.values())
    total_energy_consumption = sum(pump.get_energy_consumption() for pump in pumps.values())
    reward = - (alpha * total_flood_volume + beta * total_energy_consumption)
    return reward

def train_agent(agent, environment, actions_list, num_episodes, max_steps_per_episode):
    """
    Train the DQL agent.

    Parameters:
    - agent: Instance of DQLAgent.
    - environment: Instance of Environment.
    - actions_list: List of possible actions.
    - num_episodes: Number of training episodes.
    - max_steps_per_episode: Maximum steps per episode.

    Returns:
    - metrics: Dictionary containing training metrics.
    """
    metrics = {
        'total_rewards': [],
        'average_flood_volume': [],
        'total_energy_consumption': [],
        'epsilon_values': []
    }
    visualization = Visualization(environment.zones)

    for episode in range(num_episodes):
        environment.reset()
        state = environment.get_state()
        total_reward = 0.0
        total_flood_volume = 0.0
        total_energy = 0.0

        for step in range(max_steps_per_episode):
            # Select action index
            action_index = agent.select_action(state)
            # Retrieve action dict from actions_list
            action = actions_list[action_index]
            # Apply action to the environment
            environment.step(action)
            # Get next state
            next_state = environment.get_state()
            # Compute reward
            reward = compute_reward(environment.zones, environment.pumps)
            # Check if done
            done = (step == max_steps_per_episode - 1)
            # Store experience
            agent.memory.push(state, action_index, reward, next_state, done)
            # Optimize the model
            agent.optimize_model()
            # Update state
            state = next_state
            # Accumulate rewards and metrics
            total_reward += reward
            total_flood_volume += sum(
                max(zone.current_water_level - zone.flood_threshold, 0.0) for zone in environment.zones.values()
            )
            total_energy += sum(pump.get_energy_consumption() for pump in environment.pumps.values())

            # Retrieve current rainfall
            rainfall = environment.get_current_rainfall()

            # Update visualization every 10 steps
            if step % 10 == 0:
                visualization.update(rainfall)
                time.sleep(0.1)  # Add a 100ms delay to allow rendering

        # Update target network
        agent.update_target_network()

        # Calculate average metrics for the episode
        avg_flood_volume = total_flood_volume / max_steps_per_episode
        avg_energy_consumption = total_energy / max_steps_per_episode
        # Epsilon is calculated based on steps_done
        epsilon = agent.epsilon_final + (agent.epsilon_start - agent.epsilon_final) * \
                  np.exp(-1. * agent.steps_done / agent.epsilon_decay)

        # Log metrics
        metrics['total_rewards'].append(total_reward)
        metrics['average_flood_volume'].append(avg_flood_volume)
        metrics['total_energy_consumption'].append(avg_energy_consumption)
        metrics['epsilon_values'].append(epsilon)

        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}, "
              f"Avg Flood Volume: {avg_flood_volume:.2f}, Epsilon: {epsilon:.4f}")

        # Save the model every 50 episodes
        if (episode + 1) % 50 == 0:
            if not os.path.exists('models'):
                os.makedirs('models')
            agent.save_model(f'models/agent_episode_{episode+1}.pth')

    visualization.close()
    return metrics

def plot_performance(metrics):
    """
    Plot training performance metrics.

    Parameters:
    - metrics: Dictionary containing training metrics.
    """
    episodes = range(1, len(metrics['total_rewards']) + 1)
    plt.figure(figsize=(15, 10))

    # Total Reward
    plt.subplot(2, 2, 1)
    plt.plot(episodes, metrics['total_rewards'], label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()

    # Average Flood Volume
    plt.subplot(2, 2, 2)
    plt.plot(episodes, metrics['average_flood_volume'], label='Avg Flood Volume', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Average Flood Volume (m³)')
    plt.title('Average Flood Volume per Episode')
    plt.legend()

    # Total Energy Consumption
    plt.subplot(2, 2, 3)
    plt.plot(episodes, metrics['total_energy_consumption'], label='Total Energy Consumption', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Total Energy Consumption (units)')
    plt.title('Total Energy Consumption per Episode')
    plt.legend()

    # Epsilon Decay
    plt.subplot(2, 2, 4)
    plt.plot(episodes, metrics['epsilon_values'], label='Epsilon', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay over Episodes')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Initialize environment
    delta_time = 1.0  # Time step in seconds
    environment = Environment(delta_time=delta_time)
    
    # Generate discrete actions
    actions_list = generate_discrete_actions(environment.pumps, environment.gates)
    print(f"Total Actions Generated: {len(actions_list)}")  # Should print 504
    
    # Define state and action sizes
    state_size = len(environment.get_state())
    action_size = len(actions_list)
    
    # Initialize agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    agent = DQLAgent(state_size=state_size, device=device, actions_list=actions_list)
    
    # Define training parameters
    num_episodes = 1000
    max_steps_per_episode = 100
    
    # Train the agent
    metrics = train_agent(agent, environment, actions_list, num_episodes, max_steps_per_episode)
    
    # Plot the performance metrics
    plot_performance(metrics)
