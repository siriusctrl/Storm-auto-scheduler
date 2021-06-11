import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from Topology import Topology
from Machine import Machine

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
        return self.once()

    def reset(self):
        self.topology.reset_assignments()
        self.topology.round_robin_init()
        return self.once()
    
    def once(self):
        self.topology.update_states(time=1000, track=False, debug=True)
        

    def seed(self):
        # the np_random is the numpy RandomState
        self.np_random, seed = seeding.np_random(self.random_seed)
        # the return the seed is the seed we specify
        return [seed]

    def build_topology(self, debug=False):
        exe_info = {
            'spout':['spout', 2, [1e3, 1e3]],
            'WordCount':['bolt', 3, {}],
            'Database':['bolt', 3, {'processing_speed':60}],
            'graph': [  
                        # we first define a list of nodes
                        ['spout', 'WordCount', 'Database'], 
                        # then, we have edge represent in tuples
                        ('spout', 'WordCount'),
                        ('WordCount', 'Database')
                    ]
        }

        self.topology = Topology(2, exe_info)
        self.topology.build_executors()
        self.topology.build_homo_machines()
        self.topology.build_machine_graph([(0,1,2.)])
        self.topology.round_robin_init()

        if debug:
            # self.topology.draw_machines()
            # self.topology.draw_executors()
            print(self.topology.executor_graph.edges(data=True))
            print(self.topology.machine_to_executors)
            print(self.topology.executor_to_machines)


if __name__ == '__main__':
    env = WordCountingEnv()
    env.build_topology(debug=False)
    env.once()