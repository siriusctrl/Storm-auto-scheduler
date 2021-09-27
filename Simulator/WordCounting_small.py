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
                       n_spouts  = 5,
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
        self.edge_batch = 40

        self.seed()
        self.build_topology()

        size = len(self.topology.executor_groups)*n_machines
        self.action_space = Box(low=0.001, high=1., shape=(size,))
        # TODO: we assume fixed data incoming rate here
        ob_low = np.array([0.]*size + [0.]*n_spouts)
        ob_high = np.array([1.]*size + [20.]*n_spouts)
        self.observation_space = Box(low=ob_low, high=ob_high)
        self.topology.assignment_cache = np.zeros(((len(self.topology.executor_groups)), self.n_machines))
    
    def step(self, new_assignments):
        assert(new_assignments is not None)
        # make sure assigment for each type of bolt sum to approximately 1
        reshaped_assignments = new_assignments.reshape(((len(self.topology.executor_groups)), self.n_machines)).clip(0.001, 1.)
        # new_assignments = softmax(new_assignments, axis=1)
        total = reshaped_assignments.sum(axis=1)
        reshaped_assignments = (reshaped_assignments.T / total).T
        self.topology.update_assignments(reshaped_assignments)
        self.warm()
        metrics = self.once()
        metrics['cur'] = np.array(self.topology.assignment_cache)
        # metrics['reschedule_cost'] = np.linalg.norm(np.array(self.ac) - np.array(self.topology.assignment_cache))
        # the observation is the current deployment(after normalized) + data incoming rate
        new_state = reshaped_assignments.flatten()
        new_state = np.concatenate((new_state, metrics['avg_incoming_rate']))
        # NOTICE: this is different than the original paper where new_state is the state after normalization
        return new_state, metrics['latency'], False, metrics

    def reset(self):
        self.topology.reset_assignments()
        # self.topology.round_robin_init(shuffle=True)
        # return self.once()
        random_action = self.action_space.sample()
        return self.step(random_action)
    
    def once(self):
        return self.topology.update_states(time=1000, track=True)
    
    def warm(self, time=5000):
        self.topology.update_states(time=time, track=False)

    def seed(self):
        # the np_random is the numpy RandomState
        self.np_random, seed = seeding.np_random(self.random_seed)
        # np.random.seed(seed)
        # the return the seed is the seed we specify
        return [seed]

    def build_topology(self, debug=False): 
        all_spout = [
                {   "rate_sampler":PoissonSampler(10.), 
                    "batch":40,
                    "random_seed":self.random_seed,
                    "subset":True,
                } for _ in range(self.n_spouts)]

        exe_info = {
            'spout': ['spout', self.n_spouts, all_spout],
            'WordCount': ['bolt', 25, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':40,
                    'random_seed':None,
                }],
            'Database': ['bolt', 20, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':40,
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
        # selection = np.random.choice(list(range(len(edges))), size=(self.n_machines*(self.n_machines-1))//2, replace=False)
        # for i in selection:
        #     s, d, _, da = edges[i]
        #     edges[i] = (s, d, 40, da)

        print(edges)

        self.topology = Topology(self.n_machines, exe_info, random_seed=self.random_seed)
        self.topology.build_executors()
        self.topology.build_homo_machines(0.8)
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
    ac = env.action_space.high.reshape((3, env.n_machines))
    # ac[:,-1] = 0
    # ac[0:1,0] = 10
    # ac[1:2,1] = 10
    # ac[2:3,2] = 10
    # ac[:,0] = 10
    # print(ac)
    print(env.step(ac))
    # env.step(ac)
    for _ in range(5):
        # print(env.once())
        ac = env.action_space.sample().reshape((3, env.n_machines))
        print(env.step(ac))
