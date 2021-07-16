import numpy as np
import random
import simpy
from simpy import Environment
from collections import deque

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
                d_transform,
                batch=100,
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
            topology: Topology
                The current Topology instance
            d_transform: Object
                The pre-defined data transformation rule
            batch: int
                Define how many tuples we can process at once
                WARN: This value must be greater or equal to 1
            grouping: str
                It defines how we select the next bolt.
                current support shuffle grouping
                TODO: add support for field grouping
            random_seed: int
                random seed for reproducibility
        """
        self.id = id
        self.name = name
        self.env = env
        self.topology = topology
        self.d_transform = d_transform
        # define what is a processing speed
        self.grouping = grouping

        # make sure batch is in correct data range 
        assert(batch >= 1)
        self.batch = batch

        self.working = False
        self.queue = []
        self.action = env.process(self.run())

        self.downstreams = None

        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            self.random_seed = random_seed
        
    def run(self):
        # The bolt will run forever
        while True:
            if len(self.queue) == 0:
                self.working = False

                try:
                    if Config.debug:
                        print(f'{self} is waiting for job at {self.env.now}')
                    yield self.env.timeout(100)
                except simpy.Interrupt:
                    if Config.debug:
                        if len(self.queue) == 0:
                            print(f'{self} get interrupted while waiting at {self.env.now}')
                        else:
                            print(f'{self} get job at {self.env.now}')
            else:
                self.working = True

                if self.downstreams is None:
                    self.downstreams = self.topology.get_downstreams(self)
                    if Config.debug:
                        print(f'{self} has downstreams {self.downstreams}')

                # We need to first requesting the resource from the underlying machine
                m = self.topology.executor_to_machines[self]

                try:
                    with m.cpu.request() as req:
                        # waiting for resource acquisition to success
                        if Config.debug:
                            print(f'{self} is waiting for resources')
                        yield req
                        # the resources has been acquired from here
                        
                        processing_speed = m.capacity * m.standard

                        pdata_list = []
                        cum_time = 0
                        psize = min(self.batch, len(self.queue))
                        for i in range(psize):
                            job = self.queue[i]
                            new_data_list, pt = self.d_transform.perform(job, processing_speed)
                            cum_time += pt
                            pdata_list += new_data_list

                        # if Config.debug or Config.bolt:
                        #     print(f'{self} is processing {psize} job at {self.env.now} last {cum_time}')
                        #     print(f'{self.queue} before processing')

                        # the job processing will take place here
                        yield self.env.timeout(cum_time)

                        if Config.debug or Config.bolt:
                            print(f'{self} {self.queue} {len(self.queue)} before batch slicing')
                        # remove the processed data from the queue
                        self.queue = self.queue[psize:]

                        if Config.debug or Config.bolt:
                            print(f'{self} {self.queue} {len(self.queue)} after batch slicing')

                        if job.tracked and (len(new_data_list) > 1):
                            # we are breaking the tracked data into more pieces
                            # TODO: we should track all the intermediate generation of data
                            pass
                        
                        assert(len(pdata_list) == psize)

                        for data in pdata_list:
                            if self.downstreams == []:
                                # this is the end bolt on topology, do some wrap up
                                if Config.debug or Config.bolt:
                                    print(f'End bolt {self} finish a task {data} at {self.env.now}')

                                data.finish_time = self.env.now
                                if data.tracked:
                                    self.topology.record(data)
                            else:
                                # TODO: define generic rule for output destination
                                destination = np.random.choice(self.downstreams)
                                data.target = destination
                                data.source = self

                                bridge = self.topology.get_network(self, destination)
                                bridge.queue.append(data)
                                if Config.debug or Config.bolt:
                                    print(f'{self} sending data to {destination} at {self.env.now}')
                                
                                if (not bridge.working) and (len(bridge.queue) == 1):
                                    bridge.action.interrupt()
                except simpy.Interrupt:
                    if Config.debug or Config.update_flag:
                        print(f'{self} get interrrupted while doing job at {self.env.now}')

    def clear(self):
        """
        clear the tuple and stop doing new task
        """
        if Config.update_flag or Config.debug:
            print(f'{self} is clearing')
        self.queue = []
        self.action.interrupt()

    def __repr__(self) -> str:
        return self.to_red(f'{self.name}{self.id}')
    
    @staticmethod
    def to_red(s):
        return f"\033[91m{s}\033[00m"

if __name__ == '__main__':
    b = Bolt('test', 1)
    print(b)