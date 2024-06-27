import gym
import numpy as np
from maze_env import SimpleMazeEnv
from time import sleep

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

# Training the agent
for episode in range(100):
    state = env.reset()
    state_index = state_to_index(state)
    done = False

    while not done:
        # Choose action using epsilon-greedy policy
        randy = np.random.random()
        if randy < epsilon:
            action = env.action_space.sample()  # Exploration
        else:
            action = np.argmax(q_table[state_index]) # Exploitation: take action with highest expected payoff in current state

        next_state, reward, done, _ = env.step(action)
        next_state_index = state_to_index(next_state)

        print(next_state)
        
        # Q-table update using Bellman equation
        q_table[state_index][action] += alpha * (reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index][action])
        state_index = next_state_index

        # Visualize the maze
        env.render()
        # sleep(0.1)  # Adjust the speed of visualization

    print(f"Episode {episode + 1} completed.")

# After training, you can save a video or visualize the agent's path
