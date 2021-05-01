import numpy as np

class Topology():
    """
    A class that contain the executor connectivity and assignment information
    TODO: finish this class
    """

    def __init__(self) -> None:
        pass

    def update(self):
        """
        This method can update the interal state the executor assignments
        """
        pass

    def get_next(self, source) -> list:
        """
        get a list of downstream bolts of current source
        """
        pass

    def get_trans_delay(self, source, destination):
        """
        get the transmission delay between source and destination bolt
        """
        
        # we first retrive the host physical machine of each bolt

        # and then retrieve the trans delay between those two machines
        pass


if __name__ == '__main__':
    pass