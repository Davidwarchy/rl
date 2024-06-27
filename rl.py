import numpy as np

# Define the maze as a 2D grid (0: free space, 1: wall, 2: goal)
maze = np.array([
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 2]
])

# Define the start position
start_pos = (0, 0)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Parameters
state_size = 25  # Flattened maze (5x5 grid)
action_size = 4  # Up, down, left, right

# Create the Q-network
q_network = QNetwork(state_size, action_size)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)

import random

# Hyperparameters
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration-exploitation trade-off
epsilon_min = 0.01
epsilon_decay = 0.995

# Convert maze position to state
def get_state(maze, position):
    state = np.zeros_like(maze).flatten()
    state[position[0] * maze.shape[1] + position[1]] = 1
    return torch.FloatTensor(state)

# Get possible actions
def get_possible_actions(maze, position):
    actions = []
    if position[0] > 0 and maze[position[0] - 1, position[1]] != 1:
        actions.append(0)  # Up
    if position[0] < maze.shape[0] - 1 and maze[position[0] + 1, position[1]] != 1:
        actions.append(1)  # Down
    if position[1] > 0 and maze[position[0], position[1] - 1] != 1:
        actions.append(2)  # Left
    if position[1] < maze.shape[1] - 1 and maze[position[0], position[1] + 1] != 1:
        actions.append(3)  # Right
    return actions

# Perform the action
def take_action(position, action):
    if action == 0:
        return (position[0] - 1, position[1])
    elif action == 1:
        return (position[0] + 1, position[1])
    elif action == 2:
        return (position[0], position[1] - 1)
    elif action == 3:
        return (position[0], position[1] + 1)

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Define the goal position
goal_pos = tuple(np.argwhere(maze == 2)[0])

episode_rewards = []

# Train the agent
for episode in range(1000):
    position = start_pos
    total_reward = 0
    for t in range(100):
        state = get_state(maze, position)
        possible_actions = get_possible_actions(maze, position)

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(possible_actions)
        else:
            with torch.no_grad():
                q_values = q_network(state)
                action = possible_actions[torch.argmax(q_values[possible_actions]).item()]

        # Take action
        new_position = take_action(position, action)
        reward = -1 if maze[new_position] == 1 else (1 if maze[new_position] == 2 else 0)
        total_reward += reward

        next_state = get_state(maze, new_position)
        target = reward + gamma * torch.max(q_network(next_state)).item()
        target_f = q_network(state)
        target_f[action] = target

        # Train the network
        optimizer.zero_grad()
        loss = F.mse_loss(q_network(state), target_f)
        loss.backward()
        optimizer.step()

        position = new_position
        if reward == 1:
            break

    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    episode_rewards.append(total_reward)
    print(f"Episode {episode+1}: Total Reward: {total_reward}")

print("Training complete")

# Plot the rewards to see the learning trend
import matplotlib.pyplot as plt

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.show()  
