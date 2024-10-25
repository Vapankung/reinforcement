# train_agent.py

import numpy as np
from maze_env import MazeEnv
from dql_agent import DQNAgent

def main():
    # Define the maze sizes to train on
    maze_sizes = [4, 6, 8, 10]

    for maze_size in maze_sizes:
        print(f"\nTraining on {maze_size}x{maze_size} maze")
        # Generate a random maze
        maze = np.zeros((maze_size, maze_size), dtype=int)
        # Add random walls
        for y in range(maze_size):
            for x in range(maze_size):
                if np.random.rand() < 0.2 and not (x == 0 and y == 0) and not (x == maze_size - 1 and y == maze_size - 1):
                    maze[y][x] = 1

        env = MazeEnv(maze)
        action_size = env.action_space.n
        agent = DQNAgent(action_size)
        agent.build_model(maze.shape)

        episodes = 100  # Reduced for testing purposes
        batch_size = 32

        for e in range(episodes):
            state = env.reset()
            done = False
            time_steps = 0

            while not done:
                state_normalized = state / 2.0  # Normalize the state
                action = agent.act(state_normalized)
                next_state, reward, done, _ = env.step(action)
                next_state_normalized = next_state / 2.0  # Normalize the next state
                agent.remember(state_normalized, action, reward, next_state_normalized, done)
                state = next_state
                time_steps += 1

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            print(f"Episode {e+1}/{episodes}, Time Steps: {time_steps}, Epsilon: {agent.epsilon:.2f}")

        # Save the trained model for this maze size
        agent.save(f'dql_maze_solver_{maze_size}x{maze_size}.pth')

if __name__ == "__main__":
    main()
