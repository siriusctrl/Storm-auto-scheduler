import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

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

        self.seed()
        self.build_topology()
    
    def seed(self):
        # the np_random is the numpy RandomState
        self.np_random, seed = seeding.np_random(self.random_seed)
        # the return the seed is the seed we specify
        return [seed]
    
    def step():
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def build_topology(self):
        print('building the topology...')


if __name__ == '__main__':
    env = WordCountingEnv()