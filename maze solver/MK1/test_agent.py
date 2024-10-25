import numpy as np
from maze_env import MazeEnv
from dql_agent import DQNAgent

# Define the maze
maze = np.array([
    [0, 0, 1, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
])

env = MazeEnv(maze)
state_shape = env.observation_space.shape
action_size = env.action_space.n
agent = DQNAgent(state_shape, action_size)
agent.load('dql_maze_solver.pth')
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
