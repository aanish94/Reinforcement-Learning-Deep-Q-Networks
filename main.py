from utils import DQN, ReplayBuffer, greedy_action, epsilon_greedy, update_target, loss

import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

import gymnasium as gym
import matplotlib.pyplot as plt

# HYPERPARAMETERS
SIZE_HIDDEN_LAYERS = 64  # Size of hidden layers (# of neurons)
NUM_HIDDEN_LAYERS = 2  # Number of hidden layers
LEARNING_RATE = 0.0025 # Learning Rate
REPLAY_BUFFER_SIZE = 5000  # Replay buffer size
TOTAL_EPISODES = 500  # Number of episodes per run
EPS_INITIAL = 1.0  # Epsilon initial- exploration vs. exploitation rate. Higher F means more exploration
EPS_FINAL = 0.2  # Final epsilon value
REWARD_SCALAR = 1.0  # Reward scaling factor - stabilizes training
BATCH_SIZE = 64  # Batch size for sampling from replay memory buffer
SYNC_RATE = 20  # How often to update sync target and policy networks

# Function to calculate epsilon for a given episode
def get_epsilon(episode):
    """
    Function to calculate epsilon for a given episode

    Args:
        episode (int): current episode number
    Returns:
        epsilon (float): epsilon value for the current episode
        High epsilon means more exploration, low epsilon means more exploitation
    """

    # Linearly interpolate between epsilon_start and epsilon_end
    epsilon = max(EPS_FINAL, EPS_INITIAL - (episode / TOTAL_EPISODES) * (EPS_INITIAL - EPS_FINAL))
    # epsilon_decay_rate = 0.99
    # epsilon = max(EPS_FINAL, F * (epsilon_decay_rate ** episode))
    return epsilon

# DQN Agent class
class Agent:

    def __init__(self, env_name):
        """
        Initialize the DQN Agent
        """
        self.env_name = env_name
        self.env = gym.make(env_name)  # Create the environment
        # Define state and action space dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        # Instantiate the policy and target networks
        layers = [self.state_dim] + [SIZE_HIDDEN_LAYERS]*NUM_HIDDEN_LAYERS + [self.num_actions]
        self.policy_net = DQN(layers)  # Policy network
        # Target network is used to compute the target Q-values in the loss function
        # and provides a stable target for value function updates
        self.target_net = DQN(layers)
        update_target(self.target_net, self.policy_net)  # Sync networks
        self.target_net.eval()

        # Define the optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        # Initialize the replay buffer
        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)

    def train(self):
        """
        Train the DQN agent
        """

        steps_done = 0
        episode_durations = []

        for i_episode in range(TOTAL_EPISODES):
            epsilon = get_epsilon(i_episode)
            if (i_episode+1) % 50 == 0:
                print(f"\tEpisode {i_episode+1}/{TOTAL_EPISODES}")

            observation, _ = self.env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            while not (done or terminated):
                # Select and perform an action
                action = epsilon_greedy(epsilon, self.policy_net, state)

                observation, reward, done, terminated, info = self.env.step(action)
                reward = torch.tensor([reward]) / REWARD_SCALAR
                action = torch.tensor([action])
                next_state = torch.tensor(observation).reshape(-1).float()

                self.memory.push([state, action, next_state, reward, torch.tensor([done])])

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if len(self.memory.buffer) >= BATCH_SIZE:
                    transitions = self.memory.sample(BATCH_SIZE)
                    state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
                    # Compute loss
                    mse_loss = loss(self.policy_net, self.target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
                    # Optimize the model
                    self.optimizer.zero_grad()
                    mse_loss.backward()
                    self.optimizer.step()

                if done or terminated:
                    episode_durations.append(t + 1)

                t += 1
                steps_done += 1
                # Update the target network, copying all weights and biases in DQN
                if steps_done % SYNC_RATE == 0:
                    update_target(self.target_net, self.policy_net)

        print(f"Average return: {sum(episode_durations)/TOTAL_EPISODES}")

        return episode_durations

    def render(self):
        """
        Render the agent in the environment
        """

        render_env = gym.make(self.env_name, render_mode='human')
        reward_sum = 0
        observation, _ = render_env.reset()
        state = torch.tensor(observation).float()

        done = False
        terminated = False

        while not (done or terminated):
            render_env.render()
            # Select and perform an action
            with torch.no_grad():
                action = greedy_action(self.policy_net, state)
            observation, reward, done, terminated, _ = render_env.step(action)
            state = torch.tensor(observation).float()
            reward_sum += reward
            time.sleep(0.1)  # Add a delay to make the visualization easier to follow

        render_env.close()
        print(f"Return: {reward_sum}")
        return reward_sum

    def save_model(self, path):
        """
        Save the model to a file
        """

        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        """
        Load the model from a file
        """

        self.policy_net.load_state_dict(torch.load(path, weights_only=True))


def plot_learning_curve(runs_results):
    """
    Plot the learning curve for the agent
    """

    num_episodes = len(runs_results[0])

    results = torch.tensor(runs_results)
    means = results.float().mean(0)
    stds = results.float().std(0)

    plt.figure(figsize=(12, 9))
    plt.axhline(y=100, color='r', linestyle='--', label='Return Threshold')
    plt.plot(torch.arange(num_episodes), means, color='black', linewidth=2, label='DQN Mean Return', zorder=10)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"DQN Learning Curve over {NUM_RUNS} runs")
    plt.fill_between(np.arange(num_episodes), means - stds, means + stds, color="grey", alpha=0.7, label="Standard Deviation")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig('learning_curve.png')

def visualize_policy(agent, load_model_path=None):
    """
    Visualising the greedy Q-values for a stationary cart in the middle of the track
    2D plot showing policy as a function of pole angle and angular velocity (omega)

    This plots the policy and Q values according to the network currently
    stored in the variable "policy_net"

    Make sure to include appropriate labels and/or legends when presenting your plot
    """

    if load_model_path:
        agent.load_model('learned_model_0.pth')

    policy_net = agent.policy_net

    # Define the range of the cart states for the plot
    cart_position = 0.0  # Fix cart to middle of track
    cart_velocities = [0.0, 0.5, 1.0, 2.0]  # Iterate through a range of velocities
    angle_range = .2095 # Episode terminates at 12 degrees (0.2095 radians)
    omega_range = 1     # Angular Velocity range

    angle_samples = 100
    omega_samples = 100

    # Create a 2x2 grid of subplots
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 9))  # Policy plot
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))  # Q-value plot

    for ax1, ax2, cart_velocity in zip(axes1.ravel(), axes2.ravel(), cart_velocities):
        angles = torch.linspace(angle_range, -angle_range, angle_samples)
        omegas = torch.linspace(-omega_range, omega_range, omega_samples)

        greedy_q_array = torch.zeros((angle_samples, omega_samples))
        policy_array = torch.zeros((angle_samples, omega_samples))

        for i, angle in enumerate(angles):
            for j, omega in enumerate(omegas):
                state = torch.tensor([cart_position, cart_velocity, angle, omega])
                with torch.no_grad():
                    q_vals = policy_net(state)
                    greedy_action = q_vals.argmax()
                    greedy_q_array[i, j] = q_vals[greedy_action]
                    policy_array[i, j] = greedy_action

        # Plot policy
        contour = ax1.contourf(angles, omegas, policy_array.T, cmap='cividis')
        fig1.colorbar(contour, ax=ax1, shrink=0.8)
        ax1.set_title(f"Greedy Policy Slice for Cart with Velocity {cart_velocity}")
        ax1.set_xlabel("Pole Angle (radians)")
        ax1.set_ylabel("Pole Angular velocity (rad/s)")

        # Plot Q-values
        contour = ax2.contourf(angles, omegas, greedy_q_array.T, cmap='cividis', levels=100)
        fig2.colorbar(contour, ax=ax2, shrink=0.8)
        ax2.set_title(f"Q-function Slice for Cart with Velocity {cart_velocity}")
        ax2.set_xlabel("Pole Angle (radians)")
        ax2.set_ylabel("Pole Angular velocity (rad/s)")

    fig1.tight_layout()
    fig1.savefig('policy_plots.png')

    fig2.tight_layout()
    fig2.savefig('q_value_plots.png')


if __name__ == "__main__":
    runs_results = []
    environment = 'CartPole-v1'
    NUM_RUNS = 10

    # Store best performing agent
    best_agent = None
    best_average_return = None

    # Iterate through multiple runs
    for run in range(NUM_RUNS):
        print(f"Run {run+1}/{NUM_RUNS}:")
        # Initialize the agent and train it
        agent = Agent(environment)
        cur_returns = agent.train()
        runs_results.append(cur_returns)  # Save results for plotting
        agent.save_model(f'learned_model_{run}.pth')
        # Update best agent
        if best_agent is None:
            best_agent = agent
            best_average_return = sum(cur_returns)/TOTAL_EPISODES
        elif sum(cur_returns)/TOTAL_EPISODES > best_average_return:
            print(f"New best agent found with average return: {sum(cur_returns)/TOTAL_EPISODES}")
            best_agent = agent
            best_average_return = sum(cur_returns)/TOTAL_EPISODES

    print('Complete')

    # Plot learning curve and visualize policy
    plot_learning_curve(runs_results)
    visualize_policy(best_agent)
    best_agent.render()
