import numpy as np
from numpy.core.fromnumeric import size
from simpy import Environment
import simpy
import random
from random import choice

from Bolt import Bolt
from Config import Config
from Data import Data

class Spout():
    """
    We are assuming the spout only incure a neglectable computational cost
    """
    
    def __init__(self, id:int,
                    env:Environment,
                    topology,
                    incoming_rate:float,
                    batch:int,
                    random_seed=None
                ) -> None:
        
        self.name = 'spout'
        self.id = id
        self.incoming_rate = incoming_rate
        self.env = env

        if random_seed is not None:
            self.random_seed = random_seed
            np.random.seed(random_seed)
            random.seed(random_seed)
            self.rng = np.random.default_rng(seed=random_seed)
        else:
            self.rng = np.random.default_rng()

        self.topology = topology
        self.downstreams = None
        assert(batch >= 1)
        self.batch = batch

        self.generate_counter = 0
        self.working = False

        # I don't think anyone will interrput this process
        self.action = env.process(self.run())

        
    def run(self):
        # add this line to prevent geting downstreams before the system setup
        yield self.env.timeout(0)

        if self.downstreams is None:
            self.downstreams = self.topology.get_downstreams(self)

        # TODO: the data income rate can follow some distribution, wrap it
        interval = 1.0*self.batch / self.incoming_rate

        if Config.debug:
            print(self.__repr__(), 'interval=', interval)

        while True:
            self.working = True
            try:
                # word_list = np.random.poisson(2.7, size=self.batch)
                word_list = self.rng.poisson(2.7, size=self.batch) + 2
                # word_list = word_list[word_list > 1]
                dest:Bolt = choice(self.downstreams)
                
                new_word_list = []
                for w in word_list:
                    new = Data(w, self.env.now, f'{self.id}.{self.generate_counter}')
                    self.generate_counter += 1

                    if self.topology.tracking:
                        new.tracked = True
                        self.topology.tracking_counter += 1
                    
                    new.target = dest
                    new.source = self
                    new_word_list.append(new)

                bridge = self.topology.get_network(self, dest)

                if Config.debug:
                    if not self.topology.tracking:
                        print(self, 'generate data at', self.env.now)
                    else:
                        print(f'{self} generate tracked data at {self.env.now} with counter {self.topology.tracking_counter}')

                bridge.queue += new_word_list
                # NOTICE : interrupt is also a event, instead of function call, so the effect
                # of the state chaning (from non-working to working) will be delayed
                # without this queue == 1, the program will invoke multiple unnecessary
                # interrupt event that may cuase error and slow down the simulation
                if (not bridge.working) and (len(bridge.queue) == len(new_word_list)):
                    bridge.action.interrupt()

                yield self.env.timeout(interval)
            except simpy.Interrupt:
                if Config.update_flag or Config.debug:
                    print(f'{self} get interrupted, start again')

    def clear(self):
        if Config.update_flag or Config.debug:
            print(f'{self} is clearing')
        if self.working:
            self.action.interrupt()

    @staticmethod
    def to_yellow(prt): 
        return f'\033[93m{prt}\033[00m'

    def __repr__(self) -> str:
        return self.to_yellow(f'Spout{self.id}')

if __name__ == '__main__':
    a = Spout(1, 1e4)
    print(a)