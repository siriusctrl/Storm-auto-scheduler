import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import Actor, Critic
from memory import SequentialMemory
from random_process import OrnsteinUhlenbeckProcess
from utils import *
from ddpg import DDPG
import action_space

loss = nn.MSELoss()

class Wolptinger(DDPG):
    def __init__(self, nb_states, nb_actions, args):
        super().__init__(nb_states, nb_actions, args)

        self.knn = max(1,  int(args.max_actions * args.k_ratio))

        if args.action_space is not None:
            # we probably testing on Storm topology
            self.action_space = args.action_space
        else:
            # we are testing this on general gym problem
            self.low = args.low
            self.high = args.high
            self.action_space = action_space.Space(self.low, self.high, args.max_actions)
    
    def get_action_space(self):
        return self.action_space
    
    def select_action(self, s_t, decay_epsilon=True):
        proto_action = super().select_action(s_t, decay_epsilon=decay_epsilon)
        # print(f"Proto action: {proto_action}, proto action.shape: {proto_action.shape}")
        actions = self.action_space.search_point(proto_action, self.knn)[0]
        
        if self.knn == 1:
            self.a_t = actions[0]
            return self.a_t
        
        states = np.tile(s_t, [len(actions), 1])

        a = [to_tensor(states), to_tensor(actions)]
        # print("states: {}, actions: {}".format(a[0].size(), a[1].size()))
        actions_evaluation = self.critic(a)
        # print("actions_evaluation: {}, actions_evaluation.size(): {}".format(actions_evaluation, actions_evaluation.size()))
        actions_evaluation_np = actions_evaluation.detach().numpy()
        max_index = np.argmax(actions_evaluation_np)

        self.a_t = actions[max_index]
        return self.a_t