from flask import Flask, render_template, jsonify
import numpy as np
from maze_env import MazeEnv
from dql_agent import DQNAgent

app = Flask(__name__)

# Define the maze
maze = np.array([
    [0, 0, 1, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 1, 1, 0]
])

# Initialize the agent and load the trained model
state_shape = maze.shape
action_size = 4  # Up, Down, Left, Right
agent = DQNAgent(state_shape, action_size)
agent.load('dql_maze_solver.pth')
agent.epsilon = 0.0  # Disable exploration

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve_maze')
def solve_maze():
    env = MazeEnv(maze)
    state = env.reset()
    done = False
    path = [{'x': env.agent_pos[1], 'y': env.agent_pos[0]}]

    while not done:
        state_normalized = state / 2.0  # Normalize the state
        action = agent.act(state_normalized)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        path.append({'x': env.agent_pos[1], 'y': env.agent_pos[0]})

    return jsonify({'path': path})

if __name__ == '__main__':
    app.run(debug=True)
