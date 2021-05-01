import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class WordCountingEnv(gym.Env):

    def __init__(self, n_machines=5, 
                        max_slot  = 10,
                        max_delay = 10,
                        sigma     = 2.,
                        seed      = 20200430,
                    ) -> None:
        """
        Construct all the necessary attributes for the word couting topology
        (stream version) environment

        Parameters
        ----------
            n_machines: int
                number of physical machines that is wanted to be simulated in this environments
            max_slot: int
                maximum number of executors that each machine can hold
            max_delay: int
                The maximum transimission delay between machines
            sigma: float
                The variance of transmission delay between machines
            seed: int
                Random seed for controling the reproducibility
        """
        self.n_machines = n_machines
        self.max_slot = max_slot
        self.max_delay = max_delay
        self.sigma = sigma
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