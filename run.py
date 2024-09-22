# run.py
import gym
import numpy as np
from maze_env import SimpleMazeEnv
import json
import os

# Load the saved Q-table history and episode iteration data
file_path = os.path.join('output', 'q_learning_data.json')

with open(file_path, 'r') as f:
    data = json.load(f)

q_table_history = data['q_table_history']
q_table = np.array(q_table_history[-1])  # Load the Q-table after the last episode

# Define the maze
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0]
]

# Create the same maze environment used during training
env = SimpleMazeEnv(maze)

def state_to_index(state):
    # Convert (x, y) state tuple to a single integer index
    x, y = state
    index = x * len(maze[0]) + y
    return index

# Run the agent in the environment using the trained Q-table
total_steps = 0
state = env.reset()  # Reset the environment
state_index = state_to_index(state)
done = False

print("Running the trained model...")

while not done:
    # Choose the best action (exploitation only, no exploration)
    action = np.argmax(q_table[state_index])

    next_state, reward, done, _ = env.step(action)
    next_state_index = state_to_index(next_state)

    total_steps += 1
    state_index = next_state_index

    if done:
        print(f"Goal reached in {total_steps} steps.")
