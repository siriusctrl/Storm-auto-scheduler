from random import random
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from Topology import Topology

from Data import IdentityDataTransformer

class WordCountingEnv(gym.Env):

    def __init__(self, n_machines=4, 
                       seed      = 20200430,
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
        self.random_seed = seed

        self.topology:Topology = None

        self.seed()
        self.build_topology()
    
    def step(self, new_assignments):
        assert(new_assignments is not None)
        self.topology.update_assignments(new_assignments)
        self.warm()
        return self.once()

    def reset(self):
        self.topology.reset_assignments()
        self.topology.round_robin_init()
        return self.once()
    
    def once(self):
        return self.topology.update_states(time=1000, track=True)
    
    def warm(self):
        self.topology.update_states(time=2000, track=False)

    def seed(self):
        # the np_random is the numpy RandomState
        self.np_random, seed = seeding.np_random(self.random_seed)
        # the return the seed is the seed we specify
        return [seed]

    def build_topology(self, debug=False):
        exe_info = {
            'spout': ['spout', 2, [
                {"incoming_rate":20, "batch":100}]*2
            ],
            'SplitSentence': ['bolt', 5, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':100,
                    'random_seed':None,
                }],
            'WordCount': ['bolt', 5, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':100,
                    'random_seed':None,
                }],
            'Database': ['bolt', 5, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':100,
                    'random_seed':None,
                }],
            'graph': [
                # we first define a list of nodes
                ['spout', 'SplitSentence', 'WordCount', 'Database'],
                # then, we have edge represent in tuples
                ('spout', 'SplitSentence'),
                ('SplitSentence', 'WordCount'),
                ('WordCount', 'Database')
            ]
        }

        machine_size = 5
        bandwidth = 1e4
        batch = 100
        edges = self.build_homo_edge(machine_size, bandwidth, batch)

        self.topology = Topology(machine_size, exe_info, random_seed=20210723)
        self.topology.build_executors()
        self.topology.build_homo_machines()
        self.topology.build_machine_graph(edges)
        self.topology.round_robin_init()

        if debug:
            self.topology.draw_machines()
            self.topology.draw_executors()
            print(self.topology.executor_graph.edges(data=True))
            print(self.topology.machine_to_executors)
            print(self.topology.executor_to_machines)

    def build_homo_edge(self, num, bandwidth, batch):
        return [(i, j, bandwidth, batch) for i in range(num) for j in range(num)]

if __name__ == '__main__':
    env = WordCountingEnv()
    env.build_topology(debug=False)
    env.warm()
    print(env.once())
    env.warm()
    print(env.once())