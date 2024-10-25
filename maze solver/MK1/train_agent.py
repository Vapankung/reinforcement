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

episodes = 1000
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

agent.save('dql_maze_solver.pth')
