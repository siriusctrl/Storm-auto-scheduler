from Bolt import Bolt


class Data():

    def __init__(self, size:int, enter_time:int, tracked=False, target:Bolt=None) -> None:
        self.size = size
        self.enter_time = enter_time
        self.finish_time = None
        # this will have an id to represent a data that we sample from the 
        # simulator to give a reward to the agent.
        # None if we are not tracking this group of data
        self.tracked = tracked
        self.target = target
        self.source = None
        
    def __repr__(self):
        return f'data {self.size} {self.enter_time} s={self.source} t={self.target} {self.tracked}'