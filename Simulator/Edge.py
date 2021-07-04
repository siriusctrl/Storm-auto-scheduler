import simpy
from simpy import Environment
from Config import Config
from Data import Data

class Edge():
    """
    The is the network edge object for easier handling network update
    
    """

    def __init__(self, env:Environment) -> None:        
        self.env = env
        # key is either an executor object
        self.queue = []
        # a bandwidth represent how many byte of data can be tranfer for every milisecond
        # use 0 to represent no delay
        self.bandwidth = 0
        self.working = False
        self.debug = Config.debug
        self.action = env.process(self.run())

    def run(self):
        while True:
            if len(self.queue) == 0:
                self.working = False

                try:
                    if self.debug:
                        print('network is waiting for data')
                    yield self.env.timeout(100)
                except simpy.Interrupt:
                    if self.debug:
                        print('get something to send')
            else:
                self.working = True
                # TODO: Set capacity here later
                data:Data = self.queue.pop(0)
                
                if self.bandwidth == 0:
                    duration = 0
                else:
                    duration = data.size / self.bandwidth

                yield self.env.timeout(duration)
                # the trans end here

                target = data.target
                target.queue.append(data)
                if not target.working:
                    target.action.interrupt()
                
                if self.debug:
                    print('sent one data at', self.env.now)
    
    def __repr__(self) -> str:
        return f'{self.queue}'