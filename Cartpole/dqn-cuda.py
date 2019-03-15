
import gym
import random
import argparse
import torch
import collections
import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.nn import functional as F
from itertools import count
from torch.autograd import Variable


parser = argparse.ArgumentParser(description = "PyTorch Testing")
parser.add_argument('--no-cuda', action = 'store_true', default = False, help = 'disables cuda training')

args = parser.parse_args()


use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda' if use_cuda else 'cpu')
print(device)


class DQN(nn.Module):
	def __init__(self, input_size, output_size):
		super(DQN, self).__init__()

		self.input_size = input_size
		self.output_size = output_size
		hidden = 256

		self.run = nn.Sequential(
			nn.Linear(self.input_size, hidden),
			nn.ReLU()
		)

		self.run2 = nn.Sequential(
			nn.Linear(hidden, self.output_size)
		)

	def forward(self, x):
		x = x.cuda() if use_cuda else x

		x = self.run(x)
		x = self.run2(x)
		return x


class DQNagent():
	def __init__(self, input, output):
		self.totalMem = 10000
		self.memory = collections.deque(maxlen = self.totalMem)  # replay memory
		self.gamma = 0.80
		self.epsilon = 0.5
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.999
		self.model = DQN(input, output).to(device)  # action value function Q
		self.target = DQN(input, output).to(device)  # target action-value function Q_hat


	def remember(self, state, action, reward, next_state):
		if (self.getLength() < self.totalMem):
			data = Transition(state, action, reward, next_state)
			self.memory.append(data)

	# with epsilon probability select a random action a_t
	# else, choose action that maximizes the q function
	def act(self, state):
		sample = random.random()
		self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

		if (sample > self.epsilon):
			return self.model(state.type(torch.FloatTensor)).data.max(1)[1].view(1, 1)
		else:
			return torch.LongTensor([[random.randrange(2)]])


	def sample(self, batch_size):
		if (self.getLength() >= batch_size):
			sample = random.sample(list(self.memory), batch_size)
			return sample


	def getLength(self):
		return len(self.memory)


	def replay(self, batch_size, optimizer):
		if (self.getLength() < batch_size):
			return

		# sample random minibatch of experiences
		transitions = self.sample(batch_size)
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state))).to(device)

		non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None])).to(device)

		state_batch = Variable(torch.cat(batch.state)).to(device)
		action_batch = Variable(torch.cat(batch.action)).to(device)
		reward_batch = Variable(torch.cat(batch.reward)).to(device)

		state_action_values = self.model(state_batch).gather(1, action_batch).to(device)

		next_state_values = Variable(torch.zeros(batch_size).type(torch.Tensor)).to(device)
		next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()

		expected_state_action_values = ( (next_state_values * self.gamma) + reward_batch ).to(device)

		loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()



def main():
	max_episodes = 1000
	batch_size = 32
	episode_durations = []
	target_step = 2
	penalty = -200

	for i_episode in range(max_episodes):
		# get initial state s
		state = env.reset()
		state = torch.FloatTensor([state]).to(device)

		for i in count():
			action = agent.act(state).to(device)
			observed, reward, done, _ = env.step(action.item())
			next_state = torch.FloatTensor([observed]).to(device)
			reward = torch.FloatTensor([reward]).to(device)

			# store experiences
			if (not done):
				agent.remember(state, action, next_state, reward)
			else: # done
				agent.remember(state, action, next_state, reward + penalty)

			state = next_state
			agent.replay(batch_size, optimizer)

			if (done):
				episode_durations.append(i+1)
				print("Reward", i_episode,  ": ", i+1)
				# plot_durations(episode_durations)
				break

		if (i_episode % target_step == 0):
			agent.target.load_state_dict(agent.model.state_dict())

		mean_score = np.mean(episode_durations[-100:])
		if mean_score > env.spec.reward_threshold:
			print("Solved after {} episodes! Running average is now {}. Last episode ran to {} time steps."
				  .format(i_episode, mean_score, i))
			break

	# final_plot_durations(episode_durations)



########################################################

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	Transition = collections.namedtuple('Transition',
							('state', 'action', 'next_state', 'reward'))

	agent = DQNagent(env.observation_space.shape[0], env.action_space.n)
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.model.parameters()), lr=0.005)

	main()
