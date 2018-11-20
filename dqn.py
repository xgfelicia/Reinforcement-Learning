# code is based off "Implementing the Deep Q-Network"
# Roderick; MacGlashan; Tellex
# 2017.

# algorithm: deep q learning with experence replay

import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import collections
import math


###################################################################
# DQN

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()

		hidden = 256

		self.run = nn.Sequential(
			nn.Linear(4, hidden),
			nn.ReLU()
		)

		self.run2 = nn.Sequential(
			nn.Linear(hidden, 2),
			nn.ReLU()
		)

	def forward(self, x):
		x = self.run(x)
		x = self.run2(x)
		return x

#########################################################################
# DNQagent

class DQNagent():
	def __init__(self):
		self.totalMem = 10000
		self.memory = collections.deque(maxlen = self.totalMem)  # replay memory
		self.gamma = 0.80
		self.epsilon = 0.5
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.9999
		self.model = DQN()  # action value function Q
		self.target = DQN()  # target action-value function Q_hat


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
		non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

		non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]))

		state_batch = Variable(torch.cat(batch.state))
		action_batch = Variable(torch.cat(batch.action))
		reward_batch = Variable(torch.cat(batch.reward))

		state_action_values = self.model(state_batch).gather(1, action_batch)

		next_state_values = Variable(torch.zeros(batch_size).type(torch.Tensor))
		next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()

		expected_state_action_values = (next_state_values * self.gamma) + reward_batch

		# loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
		loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()



#############################################################################
# PLOTTING FUNCTIONS

def plot_durations(current_ep):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(current_ep, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Take 30 episode averages and plot them too
    if len(durations_t) >= 30:
        means = durations_t.unfold(0, 30, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(29), means))
        plt.plot(means.numpy())

	# pause a bit so that plots are updated
    plt.pause(0.001)


def final_plot_durations(current_ep):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(current_ep, dtype=torch.float)
    plt.title('Results')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Take 30 episode averages and plot them too
    if len(durations_t) >= 30:
        means = durations_t.unfold(0, 30, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(29), means))
        plt.plot(means.numpy())

    plt.show()


#######################################################################
# MAIN TRAINING

def main():
	max_episodes = 200
	batch_size = 32
	episode_durations = []
	target_step = 2
	penalty = -100

	for i_episode in range(max_episodes):
		# get initial state s
		state = env.reset()
		state = torch.FloatTensor([state])

		for i in count():
			action = agent.act(state)
			observed, reward, done, _ = env.step(action.item())
			next_state = torch.FloatTensor([observed])
			reward = torch.FloatTensor([reward])

			# store experiences
			if (not done):
				agent.remember(state, action, next_state, reward)
			else: # done
				agent.remember(state, action, next_state, reward + penalty)

			state = next_state
			agent.replay(batch_size, optimizer)


			if (done or i > 300):
				episode_durations.append(i+1)
				print("Duration", i_episode, ": ", i+1)
				plot_durations(episode_durations)
				break

		if (i_episode % target_step == 0):
			agent.target.load_state_dict(agent.model.state_dict())


	final_plot_durations(episode_durations)



if __name__ == '__main__':
	env = gym.make('CartPole-v0').unwrapped
	Transition = collections.namedtuple('Transition',
	                        ('state', 'action', 'next_state', 'reward'))

	agent = DQNagent()
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.model.parameters()), lr=0.001)

	main()
