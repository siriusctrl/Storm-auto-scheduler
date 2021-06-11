import numpy as np
from simpy import Environment
from Bolt import Bolt
from Config import Config
from Data import Data

class Spout():
    """
    We are assuming the spout only incure a neglectable computational cost
    """
    
    def __init__(self, id:int,
                    incoming_rate:float,
                    env:Environment,
                    topology:'topology',
                    random_seed=None
                ) -> None:
        
        self.name = 'spout'
        self.id = id
        self.incoming_rate = incoming_rate
        self.env = env

        if random_seed is not None:
            self.random_seed = random_seed
            np.random.seed(random_seed)

        self.topology = topology
        self.debug = Config.debug
        self.downstreams = None

        # I don't think anyone will interrput this process
        self.action = env.process(self.generate())

        
    def generate(self):
        yield self.env.timeout(0)

        if self.downstreams is None:
            self.downstreams = self.topology.get_downstreams(self)
        interval = 1.0 / self.incoming_rate

        if self.debug:
            print(self.__repr__(), 'interval=', interval)

        while True:

            dest_list:Bolt = np.random.choice(self.downstreams, size=self.incoming_rate, replace=True)
            word_list = np.random.randint(2, 20, size=self.incoming_rate)
            if self.debug:
                print('we generate destination list of', len(dest_list))
            
            for i in range(len(dest_list)):
                # TODO: add tracking info
                new = Data(word_list[i], self.env.now)
                new.target = dest_list[i]
                bridge = self.topology.get_network(self, dest_list[i])

                if self.debug:
                    print(self.__repr__(), 'generate data at', self.env.now)

                bridge.queue.append(new)
                if not bridge.working:
                    bridge.action.interrupt()
                
                yield self.env.timeout(interval)
                    
    def to_yellow(self, prt): 
        return f'\033[93m {prt}\033[00m'

    def __repr__(self) -> str:
        return self.to_yellow(f'Spout{self.id}')


if __name__ == '__main__':
    a = Spout(1, 1e4)
    print(a)