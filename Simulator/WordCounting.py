from random import random
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.spaces import Box

import numpy as np
from scipy.special import softmax
from Topology import Topology

from Data import IdentityDataTransformer

class WordCountingEnv(gym.Env):

    def __init__(self, n_machines= 10,
                       n_spouts  = 20,
                       data_incoming_rate = 5.,
                       seed      = 20210723,
                    ) -> None:
        """
        Construct all the necessary attributes for the word couting topology
        (stream version) environment

        Parameters
        ----------
            n_machines: int
                number of physical machines that are wanted to be simulated in this environments
            seed: int
                Random seed for controling the reproducibility
        """
        # TODO: finish this two

        self.n_machines = n_machines
        self.n_spouts = n_spouts
        self.random_seed = seed

        self.data_incoming_rate = data_incoming_rate
        self.topology:Topology = None
        self.bandwidth = 100
        self.edge_batch = 100

        self.action_space = Box(low=0.001, high=1., shape=(3*n_machines,))
        size = 3*n_machines
        # TODO: we assume fixed data incoming rate here
        ob_low = np.array([0.]*size + [self.data_incoming_rate]*n_spouts)
        ob_high = np.array([1.]*size + [self.data_incoming_rate]*n_spouts)
        self.observation_space = Box(low=ob_low, high=ob_high)

        self.seed()
        self.build_topology()
    
    def step(self, new_assignments):
        assert(new_assignments is not None)
        # make sure assigment for each type of bolt sum to approximately 1
        if new_assignments.shape != (3, self.n_machines):
            # print('action reshaped')
            new_assignments = new_assignments.reshape((3, self.n_machines)).clip(0.001, 1.)
        # new_assignments = softmax(new_assignments, axis=1)
        # TODO: Test this normalisation
        totoal = new_assignments.sum(axis=1)
        new_assignments = (new_assignments.T / totoal).T
        # print(new_assignments)
        self.topology.update_assignments(new_assignments)
        self.warm()
        reward = self.once()
        # the observation is the current deployment(after softmax) + data incoming rate
        new_state = new_assignments.flatten()
        new_state = np.concatenate((new_state, np.array([self.data_incoming_rate]*self.n_spouts)))
        # NOTICE: this is different than the original paper where new_state is the state after softmax
        return new_state, reward, False, {'pre_action':new_assignments}

    def reset(self):
        self.topology.reset_assignments()
        # self.topology.round_robin_init(shuffle=True)
        # return self.once()
        random_action = self.action_space.sample()
        return self.step(random_action)[0]
    
    def once(self):
        return self.topology.update_states(time=1000, track=True)
    
    def warm(self):
        self.topology.update_states(time=5000, track=False)

    def seed(self):
        # the np_random is the numpy RandomState
        self.np_random, seed = seeding.np_random(self.random_seed)
        # the return the seed is the seed we specify
        return [seed]

    def build_topology(self, debug=False):
        exe_info = {
            'spout': ['spout', self.n_spouts, [
                {"incoming_rate":self.data_incoming_rate, "batch":100}]*self.n_spouts
            ],
            'WordCount': ['bolt', 40, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':100,
                    'random_seed':None,
                }],
            'Database': ['bolt', 40, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':100,
                    'random_seed':None,
                }],
            'graph': [
                # we first define a list of nodes
                ['spout', 'WordCount', 'Database'],
                # then, we have edge represent in tuples
                ('spout', 'WordCount'),
                ('WordCount', 'Database')
            ]
        }

        edges = self.build_homo_edge(self.n_machines, self.bandwidth, self.edge_batch)

        self.topology = Topology(self.n_machines, exe_info, random_seed=self.random_seed)
        self.topology.build_executors()
        self.topology.build_homo_machines()
        self.topology.build_machine_graph(edges)
        self.topology.round_robin_init(shuffle=False)

        if debug:
            self.topology.draw_machines()
            self.topology.draw_executors()
            print(self.topology.executor_graph.edges(data=True))
            print(self.topology.machine_to_executors)
            print(self.topology.executor_to_machines)

    def build_homo_edge(self, num, bandwidth, batch):
        return [(i, j, bandwidth, batch) for i in range(num) for j in range(num)]

if __name__ == '__main__':
    env = WordCountingEnv(n_machines=10, n_spouts=20, data_incoming_rate=5)
    print("data incoming rate is", env.data_incoming_rate)
    # env.warm()
    # print(env.once())
    # env.warm()
    # print(env.once())

    """
    Test Obs and Action Space
    """
    # print(env.observation_space.sample())
    # print(env.action_space.sample())

    """
    Mimic Agent Action
    """
    # obs = env.reset()
    # i = 0
    # while i < 1:
    #     action = env.action_space.sample()
    #     state, reward, _, _ = env.step(action)
    #     print(reward)
    #     print(state.shape, env.observation_space.shape, action.shape)
    #     i += 1

    """
    Test the effect of bad allocations
    """
    ac = env.action_space.high.reshape((3, env.n_machines))
    # ac[:,-1] = 0
    # ac[0:1,0] = 10
    # ac[1:2,1] = 10
    # ac[2:3,2] = 10
    # ac[:,0] = 10
    print(ac)
    print(env.step(ac))
    for _ in range(20):
        print(env.once())
