from __future__ import annotations
from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from Bolt import Bolt

class Data():

    def __init__(self, size:int, enter_time:int, id:str, tracked=False, target:Bolt=None) -> None:
        self.size = size
        self.enter_time = enter_time
        self.id = id
        self.finish_time = None
        # this will have an id to represent a data that we sample from the 
        # simulator to give a reward to the agent.
        # None if we are not tracking this group of data
        self.tracked = tracked
        self.target = target
        self.source = None
    
    def replicate(self):
        new_one = Data(self.size, self.enter_time, self.id, self.tracked, self.target)
        new_one.finish_time = self.finish_time
        new_one.source = self.source
        return new_one
        
    def __repr__(self):
        # return f'data {self.size} {self.enter_time} s={self.source} t={self.target} {self.tracked}'
        return f'data {self.id} {self.tracked}'


class BaseDataTransformer:
    def perform(self, input:Data, speed):
        """
        To perform data transformation
        NOTICE: we assume the total processing time will depend on the output size 
        """
        raise NotImplementedError


class IdentityDataTransformer(BaseDataTransformer):
    def perform(self, input:Data, speed):
        return [input], input.size/speed