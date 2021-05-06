import numpy as np

class Spout():

    def __init__(self, id:int,
                    incoming_rate:float,
                    random_seed=None
                ) -> None:
        
        self.name = 'spout'
        self.id = id
        self.incoming_rate = incoming_rate

        if random_seed is not None:
            self.random_seed = random_seed
            np.random.seed(random_seed)
        
        self.job_queue = []
    
    def forward(self, topology) -> tuple:
        """
        We now send out data to downstream bolts to represent the data that
        is emitted from here during 1 second interval
        """
        dest = topology.get_next(self)
        partial = self.incoming_rate // len(dest)
        costs = []

        for d in dest:
            d.cache = partial
            # get the transition cost here
            tran_cost = topology.get_trans_delay(self, d, partial)
            costs.append(tran_cost)
        
        return costs
    
    def to_yellow(self, prt): 
        return f'\033[93m {prt}\033[00m'

    def __repr__(self) -> str:
        return self.to_yellow(f'Spout{self.id}')


if __name__ == '__main__':
    a = Spout(1, 1e4)
    print(a)