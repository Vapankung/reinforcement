# test_agent.py

import numpy as np
from maze_env import MazeEnv
from dql_agent import DQNAgent

def main():
    # Load the maze
    maze = np.array([
        [0, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0]
    ])

    env = MazeEnv(maze)
    action_size = env.action_space.n
    agent = DQNAgent(action_size)
    input_shape = maze.shape
    model_path = f'dql_maze_solver_{input_shape[0]}x{input_shape[1]}.pth'

    agent.load(model_path, input_shape)  # Provide input_shape
    agent.epsilon = 0.0  # Disable exploration

    state = env.reset()
    done = False

    while not done:
        env.render()
        state_normalized = state / 2.0  # Normalize the state
        action = agent.act(state_normalized)
        next_state, reward, done, _ = env.step(action)
        state = next_state

    env.render()

if __name__ == "__main__":
    main()
