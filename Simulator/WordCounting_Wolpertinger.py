from random import random
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.spaces import Box

import numpy as np
from scipy.special import softmax
from Topology import Topology

from Data import IdentityDataTransformer
from Sampler import BetaSampler, PoissonSampler, IdentitySampler

class WordCountingEnv(gym.Env):

    def __init__(self, n_machines= 5,
                       n_spouts  = 20,
                       data_incoming_rate = 5.,
                       seed      = 20210723,
                       bandwidth = 100,
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

        self.n_machines = n_machines
        self.n_spouts = n_spouts
        self.random_seed = seed

        self.topology:Topology = None
        self.bandwidth = bandwidth
        self.edge_batch = 100

        self.seed()
        self.build_topology()

        # TODO: change this to wolpertinger architecture
        size = n_machines*len(self.topology.executor_flat)
        self.action_space = Box(low=0.001, high=1., shape=(size,))
        # NOTICE: we are assuming a easier fixed incoming rate here
        ob_low = np.array([0.]*size + [0.]*n_spouts)
        ob_high = np.array([1.]*size + [20.]*n_spouts)
        self.observation_space = Box(low=ob_low, high=ob_high)


    
    def step(self, new_assignments):
        # TODO: change this to suit Wolpertinger architecture
        assert(new_assignments is not None)
        # make sure assigment for each type of bolt sum to approximately 1
        assert(new_assignments.size == (len(self.topology.executor_flat)*self.n_machines))
        reshaped_assignments = new_assignments.reshape((len(self.topology.executor_flat), self.n_machines))
        self.topology.update_assignments(reshaped_assignments, 'one-hot')
        self.warm()
        metrics = self.once()
        # the observation is the current deployment(after softmax) + data incoming rate
        new_state = np.concatenate((new_assignments.flatten(), metrics['avg_incoming_rate']))
        return new_state, metrics['latency'], False, {**metrics}

    def reset(self):
        self.topology.reset_assignments()
        # self.topology.round_robin_init(shuffle=True)
        # return self.once()
        random_action = self.random_action()[0]
        return self.step(random_action)[0]
    
    def once(self):
        return self.topology.update_states(time=1000, track=True)
    
    def warm(self):
        self.topology.update_states(time=5000, track=False)

    def random_action(self):
        random_proto = np.random.randn(len(self.topology.executor_flat), self.n_machines).clip(0,1)
        action = np.zeros(random_proto.shape)
        col = np.argmax(random_proto, axis=1)
        row = np.array(range(action.shape[0]))
        action[row, col] = 1
        return action.flatten(), random_proto.flatten()

    def seed(self):
        # the np_random is the numpy RandomState
        self.np_random, seed = seeding.np_random(self.random_seed)
        # the return the seed is the seed we specify
        return [seed]

    def build_topology(self, debug=False):
        low = [
                {   "rate_sampler":IdentitySampler(3.), 
                    "batch":50,
                    "random_seed":self.random_seed+offset,
                }
                for offset in range(self.n_spouts//3)]
        high = [
                {   "rate_sampler":IdentitySampler(5.), 
                    "batch":50,
                    "random_seed":self.random_seed+offset+len(low),
                }
                for offset in range(self.n_spouts//3)]
        med = [
                {   "rate_sampler":IdentitySampler(7.), 
                    "batch":50,
                    "random_seed":self.random_seed+offset+len(low)+len(high),
                }
                for offset in range(self.n_spouts-len(low)-len(high))]

        exe_info = {
            'spout': ['spout', self.n_spouts, high+med+low],
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
        return [(i, j, bandwidth, batch) for i in range(num) for j in range(num) if i != j]

if __name__ == '__main__':
    # env = WordCountingEnv(n_machines=10, n_spouts=20, data_incoming_rate=5)
    env = WordCountingEnv()
    # print("data incoming rate is", env.data_incoming_rate)
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
    ac = env.random_action()[0]
    # ac[:,-1] = 0
    # ac[0:1,0] = 10
    # ac[1:2,1] = 10
    # ac[2:3,2] = 10
    # ac[:,0] = 10
    # print(ac)
    print(env.step(ac))
    for _ in range(1):
        print(env.once())
