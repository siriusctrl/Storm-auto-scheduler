import numpy as np
import simpy
from simpy import Environment

from Config import Config

class Bolt():
    """
    This a generic class for representing bolts in any simulated stream 
    computing topology
    """


    def __init__(self, name:str,
                id:int,
                env:Environment,
                processing_speed=50,
                grouping='shuffle',
                random_seed=20200430,
                ) -> None:
        """
        Construct all the necessary at tributes for a generic bolt in a stream
        computing system simulator

        Parameters
        ----------
            name: str
                The name of the bolt
            id: int
                The number to use in order to distinguish this bolt from other
                replicas
            processing_speed: int
                The number of tuples that a bolt can process in 1 second by utilising
                1% of CPU capacity
            grouping: str
                It defines how we select the next bolt.
                current support shuffle grouping
                todo: add support for field grouping
            random_seed: int
                random seed for reproducibility
        """
        self.id = id
        self.name = name
        self.env = env
        self.processing_speed = processing_speed
        self.random_seed = random_seed
        self.grouping = grouping

        self.working = False
        self.queue = []
        self.action = env.process(self.run())
        self.debug = Config.debug
        
    def run(self):
        # The bolt will run forever
        while True:
            if len(self.queue) == 0:
                self.working = False

                try:
                    if self.debug:
                        print(self.__repr__(), 'is waiting for job')
                    yield self.env.timeout(100)
                except simpy.Interrupt:
                    if self.debug:
                        print(self.__repr__(), 'get job at', self.env.now)
                    self.working = True
            else:
                self.working = True
                # TODO:finish how bolt is going to process data
                yield self.env.timeout(2)

    def __repr__(self) -> str:
        return self.to_red(f'{self.name}{self.id}')
    
    @staticmethod
    def to_red(s):
        return f"\033[91m {s}\033[00m"

if __name__ == '__main__':
    b = Bolt('test', 1)
    print(b)