import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

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

# Define the goal position
goal_pos = tuple(np.argwhere(maze == 2)[0])

# QNetwork class definition
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

# Hyperparameters
state_size = 25  # Flattened maze (5x5 grid)
action_size = 4  # Up, down, left, right
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration-exploitation trade-off
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
max_steps_per_episode = 100
num_episodes = 1000

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

# Manually defined training loop
def train_q_learning(q_network, optimizer, maze, start_pos, gamma, epsilon, epsilon_decay, epsilon_min, num_episodes):
    episode_rewards = []
    episode_paths = []  # To store the paths taken by the agent

    for episode in range(num_episodes):
        position = start_pos
        total_reward = 0
        episode_path = [position]

        for t in range(max_steps_per_episode):
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
            target_f = q_network(state).clone().detach()
            target_f[action] = target

            # Train the network
            optimizer.zero_grad()
            loss = F.mse_loss(q_network(state), target_f)
            loss.backward()
            optimizer.step()

            position = new_position
            episode_path.append(position)

            if reward == 1:  # Reached the goal
                break

        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        episode_rewards.append(total_reward)
        episode_paths.append(episode_path)
        print(f"Episode {episode+1}: Total Reward: {total_reward}")

    print("Training complete")
    return episode_rewards, episode_paths

# Initialize Q-network and optimizer
q_network = QNetwork(state_size, action_size)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# Train the agent
episode_rewards, episode_paths = train_q_learning(q_network, optimizer, maze, start_pos, gamma, epsilon, epsilon_decay, epsilon_min, num_episodes)

# Plot the rewards to see the learning trend
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.show()

# Visualize the agent's path through the maze
plt.figure(figsize=(5, 5))
plt.imshow(maze, cmap='gray')

for path in episode_paths:
    path = np.array(path)
    plt.plot(path[:, 1], path[:, 0], marker='o')

plt.plot(start_pos[1], start_pos[0], marker='o', color='blue', markersize=10, label='Start')
plt.plot(goal_pos[1], goal_pos[0], marker='o', color='green', markersize=10, label='Goal')
plt.legend()
plt.title('Agent Path in the Maze')
plt.xlim(-0.5, maze.shape[1] - 0.5)
plt.ylim(maze.shape[0] - 0.5, -0.5)
plt.gca().invert_yaxis()
plt.show()
