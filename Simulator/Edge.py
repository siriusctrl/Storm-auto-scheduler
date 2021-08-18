import simpy
from simpy import Environment
from Config import Config
from Data import Data

class Edge():
    """
    The is the network edge object for easier handling network update
    
    """

    def __init__(self, env:Environment, batch:int=20) -> None:        
        self.env = env
        assert(batch >= 1)
        self.batch = batch
        self.queue = []
        # a bandwidth represent how many byte of data can be tranfer for every milisecond
        # use 0 to represent no delay
        self.bandwidth = 0
        self.working = False
        self.action = env.process(self.run())
        # this will contain information about which two machines the network are connecting
        self.between = []

    def run(self):
        while True:
            if len(self.queue) == 0:
                self.working = False

                try:
                    if Config.debug or Config.edge:
                        print(f'{self} is waiting for data at {self.env.now}')
                    yield self.env.timeout(100)
                except simpy.Interrupt:
                    if Config.debug or Config.edge:
                        print(f'{self} get something to send at {self.env.now}')
            else:
                try:
                    self.working = True
                    # NOTICE : Assuming unlimited network queue heree
                    # NOTICE : Only pop when the timeout finish

                    curr_list = []
                    cum_size = 0
                    psize = min(self.batch, len(self.queue))
                    for i in range(psize):
                        data:Data = self.queue[i]
                        cum_size += data.size
                        curr_list.append(data)
                    
                    if self.bandwidth > 0:
                        yield self.env.timeout(cum_size / self.bandwidth)

                    # self.queue = self.queue[psize:]
                    # del is faster than previous line
                    del self.queue[:psize]
                    
                    for data in curr_list:
                        target = data.target
                        target.queue.append(data)
                        if (not target.working) and (len(target.queue) == 1):
                            target.action.interrupt()
                        
                        if Config.debug or Config.edge:
                            print(f'{self} sent one data at {self.env.now}')
                except simpy.Interrupt:
                    if Config.debug or Config.update_flag or Config.edge:
                        print(f'{self} get interrupted at {self.env.now}')

    def clear(self):
        """
        clear the tuple and stop doing new task
        """
        if Config.update_flag or Config.debug or Config.edge:
            print(f'{self} is clearing')
        self.queue = []
        self.action.interrupt()

    def __repr__(self) -> str:
        return self.to_Cyan(f'network {self.between[0]} {self.between[1]}')

    @staticmethod
    def to_Cyan(prt): 
        return f"\033[96m{prt}\033[00m"
    

