import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# Hyperparameters
EPISODES = 100
MAX_STEPS = 200
GAMMA = 0.99            # Discount factor
LEARNING_RATE = 0.001   # Learning rate for the optimizer
MEMORY_SIZE = 10000     # Size of the replay memory
BATCH_SIZE = 64         # Batch size for training

# Set up the CartPole environment with render mode
env = gym.make("CartPole-v1", render_mode="human")

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Main DQN Agent Class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore action space
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return np.argmax(q_values.detach().numpy())  # Exploit learned values

    def remember(self, transition):
        self.memory.push(transition)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        minibatch = self.memory.sample(BATCH_SIZE)
        
        for state, action, reward, next_state, terminated in minibatch:
            target = reward
            
            if not terminated:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * np.amax(self.model(next_state_tensor).detach().numpy())
                
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_tensor)
            target_f[0][action] = target
            
            # Train the model
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

# Training the DQN Agent with Rendering
def train_dqn_agent():
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    rewards_list = []

    for e in range(EPISODES):
        state, _ = env.reset()  # Updated reset to match new API
        total_reward = 0

        for time in range(MAX_STEPS):
            env.render()  # Render the environment

            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)  # Updated unpacking

            if terminated:
                reward = -10

            agent.remember((state, action, reward, next_state, terminated))
            agent.replay()

            state = next_state
            total_reward += reward

            if terminated:
                print(f"Episode: {e}/{EPISODES}, Score: {time}, Epsilon: {agent.epsilon:.2}")
                rewards_list.append(time)
                break

    env.close()  # Close the environment after training
    return rewards_list

# Plotting the results
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.title('DQN Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

if __name__ == "__main__":
    rewards_list = train_dqn_agent()
    plot_rewards(rewards_list)

