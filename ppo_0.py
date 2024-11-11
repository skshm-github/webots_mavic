from actor_critic import Actor, Critic
from mavic import Mavic
from controller import Robot  # type: ignore
import torch
import numpy as np
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

robot = Robot()
mavic = Mavic(robot)

# PPO Hyperparameters
state_dim = 9
action_dim = 4
actor_hidden_dim1 = 512
actor_hidden_dim2 = 512
critic_hidden_dim1 = 512
critic_hidden_dim2 = 512
actor_lr = 0.0003
critic_lr = 0.001
gamma = 0.99
tau = 0.005
epsilon_clip = 0.2
entropy_coeff = 0.01

# Initialize Actor and Critic Networks
actor = Actor(state_dim, actor_hidden_dim1, actor_hidden_dim2, action_dim)
critic = Critic(state_dim, critic_hidden_dim1, critic_hidden_dim2, 1)

# Optimizers
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

# Training Hyperparameters
num_episodes = 1000
num_steps = 150
noise_std_dev = 0.3
desired_state = np.array([0, 0, 3.0, 0, 0, 10.0, 0, 0, 0])

# TensorBoard Setup
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("runs", "drone_training_ppo", current_time)
writer = SummaryWriter(log_dir=log_dir)
logger.info(f"TensorBoard logs will be saved to: {log_dir}")


def main():
    current_episode = 0
    total_steps = 0

    while current_episode < num_episodes:
        mavic.reset()
        state = get_state()
        episode_reward = 0
        episode_experiences = []

        for step in range(num_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = actor(state_tensor)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample().numpy()
            action = np.clip(action + np.random.normal(0, noise_std_dev, size=action.shape), 0, 576)
            action[1], action[2] = -action[1], -action[2]

            mavic.set_rotor_speed(action)
            for _ in range(30 // mavic.timestep):
                mavic.step_robot()

            next_state = get_state()
            state_error = float(np.linalg.norm(next_state - desired_state))
            reward = calculate_reward(state, next_state)

            episode_experiences.append((state, action, reward, next_state, action_probs[action]))
            state = next_state
            episode_reward += reward

            writer.add_scalar("Step/Reward", reward, total_steps)
            total_steps += 1
            mavic.step_robot()

        actor_loss, critic_loss = update_networks(episode_experiences)

        writer.add_scalar("Episode/Total Reward", episode_reward, current_episode)
        writer.add_scalar("Episode/Actor Loss", actor_loss, current_episode)
        writer.add_scalar("Episode/Critic Loss", critic_loss, current_episode)
        writer.flush()

        logger.info(
            f"Episode {current_episode} | "
            f"Reward: {episode_reward:.2f} | "
            f"Actor Loss: {actor_loss:.4f} | "
            f"Critic Loss: {critic_loss:.4f}"
        )

        current_episode += 1

    writer.close()
    logger.info("Training completed. TensorBoard logs saved.")


def get_state():
    imu_values = mavic.get_imu_values()
    gps_values = mavic.get_gps_values()
    gyro_values = mavic.get_gyro_values()
    return np.concatenate([imu_values, gps_values, gyro_values])


def calculate_reward(state, next_state):
    state_error = np.linalg.norm(next_state - desired_state)
    return -state_error - 0.01 * np.sum(np.square(next_state - state))


def update_networks(episode_experiences):
    states, actions, rewards, next_states, old_action_probs = zip(*episode_experiences)
    states, actions = torch.tensor(states, dtype=torch.float32), torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    old_action_probs = torch.tensor(old_action_probs, dtype=torch.float32)

    returns, advantages = compute_returns_advantages(rewards, next_states)

    new_action_probs = actor(states).gather(1, actions.unsqueeze(1)).squeeze()
    critic_values = critic(states).squeeze()

    # Policy loss with PPO clipping
    ratios = new_action_probs / (old_action_probs + 1e-10)
    surrogate1 = ratios * advantages
    surrogate2 = torch.clamp(ratios, 1 - epsilon_clip, 1 + epsilon_clip) * advantages
    actor_loss = -torch.min(surrogate1, surrogate2).mean() - entropy_coeff * new_action_probs.mean()

    # Critic loss
    critic_loss = torch.nn.functional.mse_loss(critic_values, returns)

    # Optimize the actor and critic
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return actor_loss.item(), critic_loss.item()


def compute_returns_advantages(rewards, next_states):
    returns = []
    advantages = []
    future_return = 0
    for reward in reversed(rewards):
        future_return = reward + gamma * future_return
        returns.insert(0, future_return)

    returns = torch.tensor(returns, dtype=torch.float32)
    critic_values = critic(next_states).detach().squeeze()
    advantages = returns - critic_values

    return returns, advantages


main()

