import numpy as np

class Bolt():
    """
    This a generic class for representing bolts in any simulated stream 
    computing topology
    """


    def __init__(self, name:str,
                id:int,
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
        self.processing_speed = processing_speed
        self.random_seed = random_seed
        self.grouping = grouping

        self.cache = None
    
    def process(self, topology) -> tuple:
        """
        Perform a processing of one tuple. We are assuming this is the sampled tuple that
        we want to measure in the system.
        
        Parameters
        ----------
        topolgy: Topology()
            The current topology info

        Returns
        ----------
        tuple(float, Bolt())
            returns a tuple where first value is the time, including the processing time and
            tuple transimission time to next bolt. The second value is next bolt we are passing
            the value to, None if this is the end.
            ! the unit of time here is second
        """
        # TODO: consider adding a overloading issue here
        c_time = self.compute(topology)
        t_time, next_bolt = self.trans(topology)
        
        return c_time+t_time, next_bolt
    
    def compute(self, topology, nums=1) -> float:
        # TODO: we need to make sure that the model is aware of physical machine overloading
        return nums*1000/self.processing_speed
    
    def trans(self, topology) -> tuple:
        dest_list = topology.get_next(self)
        
        if len(dest_list) == 0:
            return 0, None

        if self.grouping == 'shuffle':
            next_exe = np.random.choice(dest_list)
            topology.get_trans_delay(self, next_exe)
        elif self.grouping == 'field':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return self.to_red(f'{self.name}{self.id}')
    
    @staticmethod
    def to_red(s):
        return f"\033[91m {s}\033[00m"

if __name__ == '__main__':
    b = Bolt('test', 1)
    print(b)