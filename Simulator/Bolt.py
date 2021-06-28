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
                topology,
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
                The number of byte that a bolt can process in 1 milisecond by 
                utilising 100% of CPU capacity
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
        self.topology = topology
        # define what is a processing speed
        self.processing_speed = processing_speed
        self.random_seed = random_seed
        self.grouping = grouping

        self.working = False
        self.queue = []
        self.action = env.process(self.run())
        
        self.debug = Config.debug
        self.downstreams = None
        
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
            else:
                self.working = True

                if self.downstreams is None:
                    self.downstreams = self.topology.get_downstreams(self)
                    if self.debug:
                        print(f'{self} has downstreams {self.downstreams}')

                # We need to first requesting the resource from the underlying machine
                m = self.topology.executor_to_machines[self]

                try:
                    with m.cpu.request() as req:
                        # waiting for resource acquisition to success
                        yield req
                        # the resources has been acquired from here
                        try:
                            job = self.queue[0]
                            processing_speed = m.capacity * m.standard
                            pt = job.size / processing_speed
                            if self.debug:
                                print(f'{self} is processing job at {self.env.now} last {pt}')
                            yield self.env.timeout(pt)
                        except simpy.Interrupt:
                            print('The job processing get interrrupt')
                        
                        # TODO: we might need to perform some data transformation here
                        data = self.queue.pop(0)
                        
                        if self.downstreams == []:
                            # this is the end bolt on topology
                            # TODO: we should do something for the ending bolt
                            pass
                        else:
                            destination = np.random.choice(self.downstreams)
                            # TODO:we should do something for the data tramsformation
                            data.target = destination
                            bridge = self.topology.get_network(self, destination)
                            bridge.queue.append(data)
                            if self.debug:
                                print(f'{self} sending data to {destination}')
                            
                            if not bridge.working:
                                bridge.action.interrupt()

                except simpy.Interrupt:
                    print('the task get interrupted while waiting for resources')

    def __repr__(self) -> str:
        return self.to_red(f'{self.name}{self.id}')
    
    @staticmethod
    def to_red(s):
        return f"\033[91m {s}\033[00m"

if __name__ == '__main__':
    b = Bolt('test', 1)
    print(b)