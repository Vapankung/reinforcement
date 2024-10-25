# maze_env.py

import gym
from gym import spaces
import numpy as np

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, maze):
        super(MazeEnv, self).__init__()
        self.maze = maze
        self.maze_height, self.maze_width = maze.shape
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.maze_height, self.maze_width), dtype=np.uint8)
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]  # Starting position
        self.goal_pos = [self.maze_height - 1, self.maze_width - 1]
        return self._get_obs()

    def step(self, action):
        # Map action to movement
        moves = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        move = moves.get(action, (0, 0))
        new_pos = [self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]]

        # Check boundaries and walls
        if 0 <= new_pos[0] < self.maze_height and 0 <= new_pos[1] < self.maze_width:
            if self.maze[new_pos[0], new_pos[1]] == 0:
                self.agent_pos = new_pos  # Move agent if no wall

        # Check if goal is reached
        done = self.agent_pos == self.goal_pos
        reward = 1 if done else -0.1  # Reward for reaching the goal

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # Return the maze with the agent's position
        obs = np.copy(self.maze)
        obs[self.agent_pos[0], self.agent_pos[1]] = 2  # Mark agent's position
        return obs

    def render(self, mode='human'):
        # Simple console output
        render_grid = self._get_obs()
        render_grid[self.goal_pos[0], self.goal_pos[1]] = 3  # Mark goal
        symbols = {0: ' ', 1: '█', 2: 'A', 3: 'G'}
        print("\n".join([
            "".join([symbols[cell] for cell in row]) for row in render_grid
        ]))
        print()
