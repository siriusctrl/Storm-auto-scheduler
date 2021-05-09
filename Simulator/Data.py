class Data():

    def __init__(self, size:int, enter_time:int, track_id=None) -> None:
        self.size = size
        self.enter_time = enter_time

        # this will have an id to represent a data that we sample from the 
        # simulator to give a reward to the agent. None if we are not
        # tracking this group of data
        self.track_id = track_id
        
        # A starting time and end time for processing/trasmitting this data
        self.start = None
        self.end = None
    
    def __repr__(self):
        return f'data {self.size} s={self.start} e={self.end}'