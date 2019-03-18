
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions


parser = argparse.ArgumentParser(description = "PyTorch DQN")
parser.add_argument("--no-cuda", action = 'store_true', default = False,
						help = 'disables cuda training')
parser.add_argument("--gamma", type = float, default = 0.99)
parser.add_argument("--fg-size", type = int, default = 100)
parser.add_argument("--batch-size", type = int, default = 20)
parser.add_argument("--max-iterations", type = int, default = 250)
parser.add_argument("--learning-rate", type = float, default = 0.001)
parser.add_argument("--epsilon", type = float, default = 0.5)


ARGS = parser.parse_args()
print(vars(ARGS))




# policy neural net class
class Policy(nn.Module):
	def __init__(self, state_size, action_size):
		super(Policy, self).__init__()
		hidden = 128

		self.run = nn.Sequential(
			nn.Linear(state_size, hidden),
			nn.Dropout(p = 0.5),
			nn.ReLU(),
			nn.Linear(hidden, action_size),
			nn.Softmax(dim = -1)
		)

		# Overall reward and loss history
		self.reward_history = []
		self.loss_history = []

		# Episode actions and rewards
		self.episode_actions = torch.Tensor([])
		self.episode_rewards = []

	# resets episode history
	def reset(self):
		self.episode_actions = torch.Tensor([])
		self.episode_rewards = []


	def forward(self, x):
		return self.run(x)


# agent class
class Agent():
	def __init__(self, state_size, action_size):
		self.model = Policy(state_size, action_size)
		self.learning_rate = ARGS.learning_rate #0.005
		self.gamma = ARGS.gamma #0.99


	def predict(self, state):
		# Policy model takes input state and outputs the probabilites of the actions
		# and choosing based on the probabilities in state
		state = torch.from_numpy(state).type(torch.FloatTensor)
		action_probs = self.model(state)

		# store as batch of relative probability
		distribution = torch.distributions.Categorical(action_probs)

		# generate sample based on distribution
		action = distribution.sample()

		# concatenate actions with log probability
		self.model.episode_actions = torch.cat([
			self.model.episode_actions,
			distribution.log_prob(action).reshape(1)
		])

		return action


	def update_policy(self):
		R = 0
		rewards = []

		# Discount future rewards back to the present using gamma
		for r in self.model.episode_rewards[::-1]:
			R = r + self.gamma * R
			rewards.insert(0, R)

		# standardize rewards with mean and sd and prevent 0 denominator
		# standardize to get relative reward
		# finfo().eps =  2.22044604925e-16
		rewards = torch.FloatTensor(rewards)
		rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float64).eps)

		# Calculate loss (multiply reward by -1 )
		loss = (torch.sum(torch.mul(self.model.episode_actions, rewards).mul(-1), -1))

		# Update network weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Save and intialize episode history counters
		self.model.loss_history.append(loss.item())
		self.model.reward_history.append(np.sum(self.model.episode_rewards))
		self.model.reset()


	def train(self, episodes, max_step):
		scores = []

		for episode in range(episodes):
			state = env.reset()

			for time in range(max_step):
				# get action using NN to predict
				action = self.predict(state)

				# Step through environment using chosen action
				state, reward, done, _ = env.step(action.item())

				# save results
				# if episode done, start new env
				self.model.episode_rewards.append(reward)
				if done:
					break

			# update policy after episode done
			self.update_policy()

			# Calculate score to determine when the environment has been solved
			scores.append(time)
			mean_score = np.mean(scores[-100:]) # last 100 times > 195.0

			if episode % 50 == 0:
				print('Episode {}\tAverage length (last 100 episodes): {:.2f}'.format(
					episode, mean_score))

			if mean_score > env.spec.reward_threshold:
				print("Solved after {} episodes! Running average is now {}. Last episode ran to {} time steps."
					  .format(episode, mean_score, time))
				break


########################################################

def visualize(policy):
	# number of episodes for rolling average
	window = 30

	fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
	rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
	std = pd.Series(policy.reward_history).rolling(window).std()
	ax1.plot(rolling_mean)
	ax1.fill_between(range(len(policy.reward_history)), rolling_mean -
	                 std, rolling_mean+std, color='orange', alpha=0.2)
	ax1.set_title(
	    'Episode Length Moving Average ({}-episode window)'.format(window))
	ax1.set_xlabel('Episode')
	ax1.set_ylabel('Episode Length')

	ax2.plot(policy.reward_history)
	ax2.set_title('Episode Length')
	ax2.set_xlabel('Episode')
	ax2.set_ylabel('Episode Length')

	fig.tight_layout(pad=2)
	plt.show()



def final_plot_rewards(policy):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(policy.reward_history, dtype=torch.float)
    plt.title('Reward Results')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Take 30 episode averages and plot them too
    if len(durations_t) >= 30:
        means = durations_t.unfold(0, 30, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(29), means))
        plt.plot(means.numpy())

    plt.show()

##############################################################

if __name__ == '__main__':
	# this environment is 'solved' if average reward > 195.0 over 100 consecutive trials
	env = gym.make('CartPole-v0')

	agent = Agent(env.observation_space.shape[0], env.action_space.n)
	optimizer = optim.Adam(agent.model.parameters(), lr = agent.learning_rate)


	agent.train(episodes = 1000, max_step = 300)
	visualize(agent.model)
	final_plot_rewards(agent.model)
