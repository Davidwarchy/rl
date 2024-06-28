import gym
import numpy as np
from maze_env import SimpleMazeEnv
from time import sleep
import json

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.9  # Exploration-exploitation trade-off

# Create a simple maze environment
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0]
]
env = SimpleMazeEnv(maze)

# Q-table initialization
q_table = np.zeros((env.observation_space.n, env.action_space.n))

def state_to_index(state):
    # Convert (x, y) state tuple to a single integer index
    x, y = state
    index = x * len(maze[0]) + y
    return index

# Training parameters
num_episodes = 100
episode_iterations = []  # To store number of iterations per episode
save_interval = 10  # Save data every 10 episodes

# Training the agent
for episode in range(num_episodes):
    state = env.reset()
    state_index = state_to_index(state)
    done = False
    iterations = 0

    while not done:
        # Choose action using epsilon-greedy policy
        randy = np.random.random()
        if randy < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            action = np.argmax(q_table[state_index])  # Exploitation

        next_state, reward, done, _ = env.step(action)
        next_state_index = state_to_index(next_state)

        # Q-table update using Bellman equation
        q_table[state_index][action] += alpha * (reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index][action])
        state_index = next_state_index

        iterations += 1

    episode_iterations.append(iterations)
    print(f"Episode {episode + 1} completed. Iterations: {iterations}")

    # Periodically save Q-table and episode iteration data
    if (episode + 1) % save_interval == 0 or episode == num_episodes - 1:
        data = {
            'q_table': q_table.tolist(),
            'episode_iterations': episode_iterations
        }

        file_path = f'q_learning_data_episode_{episode + 1}.json'
        with open(file_path, 'w') as f:
            json.dump(data, f)

        print(f"Q-table and episode iteration data saved to {file_path}")

print("Training complete.")
