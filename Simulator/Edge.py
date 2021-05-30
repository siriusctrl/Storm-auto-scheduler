
class Edge():
    """
    The is the network edge object for easier handling network update
    There are mainly three situation:
    1. The network has no queuing jobs
        - once we received a pack of data, we could process those data directly
    2. If we have some queueing jobs from the same source but no other queueing job
        - We can just send those jobs in sequence
    3. If there are many queuing jobs from different sources, we need to consider:
        1. If those jobs are coming at different time-frame, for example, job1 from source1
           staring at 1000 and job2 from source2 starting at 1050. 
                - We need to first process job1's data until 1050. 
                - Then, enter next choice
        2. If those jobs are coming at same time frame
            - we need to distributed our time budget equally to those data
            - for example, if we have 999 time bugets now and have 3 jobs from 3 different source
              to do. We need to give each source 333 to send their jobs.
    """

    def __init__(self) -> None:
        # key is either an executor object
        self.job_queue = {}

        # a weight represent the cost for communicating on this edge
        self.weight = 0

        # time budget we left for job processing
        self.budget = 0
    
    def __repr__(self) -> str:
        return f'{self.job_queue}'