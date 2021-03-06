
import gym
import random
import collections
import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable



class DQN(nn.Module):
	def __init__(self, state_size, action_size):
		super(DQN, self).__init__()
		hidden = 256
		hidden2 = 512

		self.run = nn.Sequential(
			nn.Linear(state_size, hidden),
			nn.ReLU(),
			nn.Linear(hidden, action_size)
		)

	def forward(self, x):
		x = self.run(x)
		return x

#########################################################################
# DNQagent

class DQNagent():
	def __init__(self, state_size, action_size):
		self.totalMem = 10000
		self.memory = collections.deque(maxlen = self.totalMem)
		self.gamma = 0.8
		self.epsilon = 0.5
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.999
		self.model = DQN(state_size, action_size)
		self.target = DQN(state_size, action_size)


	def remember(self, state, action, reward, next_state):
		if (self.getLength() < self.totalMem):
			data = Transition(state, action, reward, next_state)
			self.memory.append(data)


	def act(self, state ):
		sample = random.random()
		self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

		# epsilon greedy algorithm
		if (sample > self.epsilon ):
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

		# experience replay
		transitions = self.sample(batch_size)
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
											  batch.next_state)))

		non_final_next_states = Variable(torch.cat([s for s in batch.next_state
													if s is not None]))

		state_batch = Variable(torch.cat(batch.state))
		action_batch = Variable(torch.cat(batch.action))
		reward_batch = Variable(torch.cat(batch.reward))

		state_action_values = self.model(state_batch).gather(1, action_batch)
		next_state_values = Variable(torch.zeros(batch_size).type(torch.Tensor))

		# get max q-value using target parameters, which are updated every t steps
		next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()
		expected_state_action_values = (next_state_values * self.gamma) + reward_batch

		# get index of the action chosen (action selection)
		policy_index = list(expected_state_action_values).index(expected_state_action_values.max())

		# calculate Y_t using chosen action (action evaluation)
		target_result = self.target.forward(non_final_next_states[policy_index: ]).max(1)[0].detach()
		expected_target_action_value = (target_result * self.gamma) + reward_batch[policy_index: ]

		loss = F.mse_loss(state_action_values[policy_index: ].view(-1), expected_target_action_value)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()



#############################################################################


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


######################################################################


def main():
	max_episodes = 500
	batch_size = 64
	episode_durations = []
	target_update = 10
	step = 0
	penalty = -300

	for i_episode in range(max_episodes):
		state = env.reset()
		state = torch.FloatTensor([state])
		step += 1

		for i in count():
			action = agent.act(state)
			observed, reward, done, _ = env.step(action.item())
			next_state = torch.FloatTensor([observed])
			reward = torch.FloatTensor([reward])

			if (not done):
				agent.remember(state, action, next_state, reward)
			else: # done
				agent.remember(state, action, next_state, reward + penalty )

			state = next_state
			agent.replay(batch_size, optimizer)

			if (done or i > 300):
				episode_durations.append(i+1)
				print("Duration", i_episode, ": ", i+1)
				plot_durations(episode_durations)
				break

		if step % target_update == 0:
			agent.target.load_state_dict(agent.model.state_dict())

	final_plot_durations(episode_durations)


if __name__ == '__main__':
	env = gym.make('CartPole-v0').unwrapped
	Transition = collections.namedtuple('Transition',
	                        ('state', 'action', 'next_state', 'reward'))

	agent = DQNagent(env.observation_space.shape[0], env.action_space.n)
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.model.parameters()), lr=0.001)

main()
