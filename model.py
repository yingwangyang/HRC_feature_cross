import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import pandas as pd

from collections import namedtuple, deque
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
import random
import math
import torch.optim as optim

# load from config
BATCH_SIZE = 20
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100
LR = 0.01
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Net(nn.Module):
    
    def __init__(self, n_states_dim, n_actions_dim, hidden_dim=100, init_weight=0.1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states_dim + n_actions_dim, hidden_dim)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.out = nn.Linear(hidden_dim, 1)
        self.out.weight.data.normal_(-init_weight, init_weight)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value
    
class DQN(object):
    
    def __init__(self, n_states_dim, n_actions_dim, n_actions, hidden_dim, 
                       init_weight, capacity, device):
        self.eval_net = Net(n_states_dim, n_actions_dim, hidden_dim, init_weight).to(device)
        self.target_net = Net(n_states_dim, n_actions_dim, hidden_dim, init_weight).to(device)
        self.memory = deque([], maxlen=capacity)
        self.n_actions = n_actions
        self.device=device
        self.steps_done = 0
        self.optimizer = optim.AdamW(self.eval_net.parameters(), lr=LR, amsgrad=True)
        self.learn_step_counter = 0
        self.TARGET_REPLACE_ITER = 5
        self.GAMMA = GAMMA
        self.loss_func = nn.MSELoss()
        self.capacity = capacity
    
    # replay
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    # select action
    # meta controller returns top1
    # controller returns topk
    def select_action(self, state, action_to_embedding, topk):
        # np.random.uniform < epsilon
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            action_res = {}
            # forward
            for key, value in action_to_embedding.items():
                input = torch.cat((state, value), 1)
                action_value = self.eval_net(input)
                action_res[key] = action_value.tolist()[0]
            # get indices of the action value
            sorted_action_res = sorted(action_res.items(), key = lambda x: x[1], reverse=True)
            # print(sorted_action_res)
            action_list = []
            for index in range(topk):
                action_list.append(sorted_action_res[index][0])
            action = torch.tensor([action_list], device=self.device, dtype=torch.long)
            # print(action)    
            # action = torch.topk(action_value, topk, largest=True)[1].view(1, topk)
        else:
            action_list = []
            for key, value in action_to_embedding.items():
                action_list.append(key)
            action = torch.tensor([random.sample(action_list, topk)], 
                                  device=self.device, dtype=torch.long)
        return action
    
    def learn(self, action_to_embedding, logger):
        if self.__len__() < self.capacity:
            return
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        transitions = self.sample(BATCH_SIZE)
        
        batch = Transition(*zip(*transitions))
        
        # state
        state = list(batch.state)
        for index in range(BATCH_SIZE):
            action = batch.action[index].tolist()[0][0]
            select_action_embeddings = action_to_embedding[action]
            state[index] = torch.cat((state[index], select_action_embeddings), 1)
        state_batch = torch.cat(tuple(state))
        
        state_action_values = self.eval_net(state_batch)
        
        # next_state 
        next_state_values = 0
        first = True
        for action, value in action_to_embedding.items():
            next_state = list(batch.next_state)
            for index in range(BATCH_SIZE):
                next_state[index] = torch.cat((next_state[index], value), 1)
            next_state_batch = torch.cat(tuple(next_state))
            if first:
                next_state_values = self.target_net(next_state_batch)
                first = False
            else:
                next_state_values = torch.cat((next_state_values, self.target_net(next_state_batch)), 1)
        
        next_state_values = next_state_values.max(1)[0]   
        
        reward_batch = torch.cat(batch.reward)
        
        # bellman equation
        expected_state_action_values = reward_batch + (self.GAMMA * next_state_values)
        
        loss = self.loss_func(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logger.debug("loss: {}".format(loss))
        