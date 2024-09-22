import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from maze_env import SimpleMazeEnv
import json
import os

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        """
        The `replay` method trains the agent by sampling a batch of experiences from memory:
        - It updates the Q-value for each action using the Bellman equation:
        \[
        Q(s, a) = r + \gamma \max_a' Q(s', a')
        \]
        where \( r \) is the reward, \( \gamma \) is the discount factor, and \( s' \) is the next state.
        - The neural network is optimized using backpropagation to minimize the loss between the target Q-value and the predicted Q-value.
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state).detach())
            target_f = self.model(state)
            target_f[0][0][action] = target
            loss = self.criterion(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Create maze environment
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0]
]
env = SimpleMazeEnv(maze)

# Initialize DQN agent
state_size = 2  # x and y coordinates
action_size = 4  # up, down, left, right
agent = DQNAgent(state_size, action_size)

# Training parameters
num_episodes = 1000
batch_size = 32
episode_iterations = []

# Training the agent
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    iterations = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        iterations += 1

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    episode_iterations.append(iterations)
    print(f"Episode {episode + 1}/{num_episodes}, iterations: {iterations}, epsilon: {agent.epsilon:.2f}")

# Save training data
output_data = {
    'episode_iterations': episode_iterations
}

output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)
file_path = os.path.join(output_folder, 'dqn_learning_data.json')

with open(file_path, 'w') as f:
    json.dump(output_data, f)

print(f"Training data saved to {file_path}")
print("Training complete.")

# Save the trained model
torch.save(agent.model.state_dict(), os.path.join(output_folder, 'dqn_model.pth'))
print("Model saved.")