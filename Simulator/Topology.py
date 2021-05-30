import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from Bolt import Bolt
from Data import Data
from Edge import Edge
from Machine import Machine
from Spout import Spout

class Topology():
    """
    A class that contains the executor connectivity and their assignment information
    TODO: finish this class
    """

    def __init__(self,
                n_machines:int,
                executors_info:dict,
                inter_trans_delay=0.,
                random_seed:int=None,
        ) -> None:
        """
        A generic topology constructor in a stream computing system simulator

        Parameters
        ----------
        n_machines
            number of machines we have
        executors_info
            Each key is the name of the executor and the value is a list contains associated
            information. Refer to sample for more details
        random_seed
            np random seed for reproducibility
        """

        # we assume the transmission delay are identical between two same machines
        # Thus, an graph without direction is used
        self.machine_graph = nx.Graph()
        self.executor_graph = nx.DiGraph()

        self.n_machines = n_machines
        self.machine_list = []

        # key: name of the executor, value: [type of executor (in str), number of replicas]
        self.executor_info = executors_info

        self.inter_trans_delay = inter_trans_delay

        self.executor_to_machines = {}
        self.machine_to_executors = {}
        self.name_to_executors = {}

        # ! unit should be in ms
        self.universal_time = 0

        self.random_seed = random_seed

        self.track_counter = 0
        self.track_datas = {}

        # setup random seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def update_assignments(self, new_assignments):
        pass

    def update_states(self, time:int=1000, track=False, debug=False):
        """
        This method can update the interal state the simulator

        Parameters
        -----------
        time
            fast forward the system about this amount of miliseconds
        """
        added = set()
            
        update_queue = ['spout']
        
        while len(update_queue) > 0:
            if debug:
                print('************************************')
                print(f'added={added}\nupdate_queue={update_queue}')

            curr = update_queue.pop(0)
            
            if type(curr) is Bolt:
                print(f'get {curr}')
            elif type(curr) is Spout:
                curr:Spout
                d_size = curr.incoming_rate
                s_data = Data(size=d_size, enter_time=self.universal_time)

                if track:
                    s_data.track_id = self.track_counter
                    self.track_counter += 1
                
                s_data.start = self.universal_time
                s_data.end = self.universal_time + time

                self.push_data_to_next(curr, s_data, added, update_queue, debug=debug)
            elif curr in self.name_to_executors:
                update_queue = self.name_to_executors[curr] + update_queue
            else:
                if debug:
                    print(f'get an edge {curr}')
                
                if curr[-1] == True:
                    pass


    def push_data_to_next(self, source, data:Data, added:set, update_queue:list, debug=False):
        successors = self.get_downstreams(source)
        # TODO: we assume shuffle grouping here, might need to extend it later

        if type(source) is Spout:
            
            partition_size = data.size // len(successors)

            for s in successors:
                s:Bolt
                source_machine:Machine = self.executor_to_machines[source]
                target_machine:Machine = self.executor_to_machines[s]

                p_data = Data(partition_size, data.enter_time, track_id=data.track_id)
                p_data.start = data.start
                p_data.end = data.end
            
                if source_machine != target_machine:
                    
                    edge:Edge = self.machine_graph[source_machine][target_machine]['object']

                    if (source_machine, target_machine) not in added:
                        update_queue.append((source_machine, target_machine, s, True))
                        added.add((source_machine, target_machine))
                    else:
                        update_queue.append((source_machine, target_machine, s, False))
                    
                    edge.job_queue[source] = edge.job_queue.get(source, []) + [p_data]
                else:
                    # if the inter-connection cost is 0
                    if self.inter_trans_delay == 0:
                        update_queue.append((source_machine, source_machine, s, False))
                        s.job_queue[source] = s.job_queue.get(source, []) + [p_data]
                    else:
                        pass

                    if debug:
                        print(f'job_queue on {s} is {s.job_queue}')
        
        elif type(source) is Bolt:
            # TODO: to find whether there are edges or this is the end bolt
            pass

        else:
            raise ValueError(f'Unknown type to push {source}')
        
        if debug:
            print(self.machine_graph.edges(data=True))
            

    def round_robin_init(self) -> None:
        """
        Initialise the resources in a round-robin manner
        """
        assert(self.machine_list is not None)
        assert(self.machine_list != [])

        exec = []

        for e_list in self.name_to_executors.values():
            exec += e_list
        
        r = len(self.machine_list)

        for i in range(len(exec)):
            to = i % r
            self.add_executor_to_machines(exec[i], self.machine_list[to])
            self.add_machine_to_executors(exec[i], self.machine_list[to])
            
    def get_downstreams(self, source) -> list:
        """
        get a list of downstream bolts of current source
        """
        if type(source) is str:
            successors = list(self.executor_graph.successors(source))[0]
        else:
            successors = list(self.executor_graph.successors(source.name))[0]
    
        return self.name_to_executors[successors]

    def get_trans_delay(self, source, destination, data_size):
        """
        get the transmission delay between source and destination bolt
        """
        # we first retrive the host physical machine of each bolt
        source_m = self.executor_to_machines[source]
        dest_m = self.executor_to_machines[destination]
        # and then retrieve the trans delay between those two machines

        cost_per_data = self.machine_graph.get_edge_data(
                            source_m, dest_m, default=self.inter_trans_delay
                        )

        # TODO: add the job to job queue of edges
        if cost_per_data == 0:
            return 0
        else:
            return data_size*1000 / cost_per_data

    def build_machine_graph(self, edges):
        self.machine_graph.add_nodes_from(self.machine_list)

        for s, d, w in edges:
            ns = self.machine_list[s]
            nd = self.machine_list[d]
            self.machine_graph.add_edge(ns, nd)

            # initialise the edge object
            ob = Edge()
            ob.weight = w
            self.machine_graph[ns][nd]['weight'] = w
            # the job queue dict for each edge has the following 
            # key: the upstream object, value is a list that represent a FIFO queue
            # which represent all the task coming from that(the key) upstream executor
            self.machine_graph[ns][nd]['object'] = ob

    def add_executor_to_machines(self, executor, machine):
        self.executor_to_machines[executor] = machine
        
    def add_machine_to_executors(self, executor, machine):
        self.machine_to_executors[machine] = self.machine_to_executors.get(machine, []) + [executor]

    def create_spouts(self, n, data_rates):
        assert(len(data_rates) == n)

        for i in range(n):
            new_spout = Spout(i, data_rates[i], self.random_seed)
            self.name_to_executors['spout'] = self.name_to_executors.get('spout', []) + [new_spout]
  
    def create_bolts(self, n, name, **bolt_info):
        for i in range(n):
            new_bolt = Bolt(name, i, **bolt_info)
            self.name_to_executors[name] = self.name_to_executors.get(name, []) + [new_bolt]

    def create_executor_graph(self, nodes, edges):
        self.executor_graph.add_nodes_from(nodes)
        self.executor_graph.add_edges_from(edges)

    def build_executors(self):
        for name, v in self.executor_info.items():
            if v[0] == 'spout':
                self.create_spouts(v[1], v[2])
            elif v[0] == 'bolt':
                self.create_bolts(v[1], name, **v[2])
            elif name == 'graph':
                self.create_executor_graph(v[0], v[1:])
            else:
                raise ValueError('Unknown type of executor')
    
    def build_homo_machines(self, capacity=1):
        """
        Parameters
        -----------
        capacity
            build n machines all with the same computational capactiy
        """
        self.build_heter_machines(capacity_list=[capacity]*self.n_machines)

    def build_heter_machines(self, capacity_list:list):
        """
        Parameters
        -----------
        capacity_list
            build n machines with the the corresponding computational capactiy
            specified in this list
        """
        assert(len(capacity_list) == self.n_machines)
        for i in range(self.n_machines):
            m = Machine(i, capacity=capacity_list[0])
            self.machine_list.append(m)

    def build_sample(self, debug=False):
        self._build_sample_machines()
        self._build_sample_executors()
        self.round_robin_init()
        if debug:
            print(self.name_to_executors)

    def _build_sample_machines(self):
        self.n_machines = 4
        self.build_homo_machines()
        edges = [(0,1,2.), (0,2,3.), (0,3,1.5), (1,2,3.5), (1,3,2.5), (2,3,3.5)]
        self.build_machine_graph(edges)

    def _build_sample_executors(self):
        sample_info = {
            'spout':['spout', 2, [1e3, 1e3]],
            'SplitSentence':['bolt', 3, {'processing_speed':20}],
            'WordCount':['bolt', 3, {}],
            'Database':['bolt', 3, {'processing_speed':60}],
            'graph': [  
                        # we first define a list of nodes
                        ['spout', 'SplitSentence', 'WordCount', 'Database'], 
                        # then, we have edge represent in tuples
                        ('spout', 'SplitSentence'), 
                        ('SplitSentence', 'WordCount'), 
                        ('WordCount', 'Database')
                    ]
        }

        self.executor_info = sample_info
        
        self.build_executors()

    def draw_machines(self):
        pos = nx.kamada_kawai_layout(self.machine_graph)
        # draw edges and weights
        labels = nx.get_edge_attributes(self.machine_graph, 'weight')
        # draw the graph
        nx.draw(self.machine_graph, pos, with_labels=True)
        # draw the edge labels
        nx.draw_networkx_edge_labels(self.machine_graph, pos, edge_labels=labels)
        plt.show()
    
    def draw_executors(self):
        pos = nx.spring_layout(self.executor_graph)
        nx.draw_networkx(self.executor_graph, pos)
        plt.show()
    
    def reset_assignments(self):
        self.machine_to_executors = {}
        self.executor_to_machines = {}
        # TODO: should clear any suspending jobs in egde and bout/spout


if __name__ == '__main__':
    test = Topology(4, {})
    test.build_sample(debug=False)

    """
    Graph example
    """
    # for i in test.machine_list:
    #     print(i.capacity)
    # test.draw_machines()
    # test.draw_executors()

    """
    Name to executor example
    """
    # print(test.name_to_executors['SplitSentence'][0].processing_speed)

    """
    Get next example
    """
    # print(test.get_next('SplitSentence'))

    """
    Test Round Robin init
    """
    # print(test.machine_to_executors)
    # print(test.executor_to_machines)
    # test.round_robin_init()
    # print(test.machine_to_executors)
    # print(test.executor_to_machines)

    test.update_states()
    # print(test.topology.executor_graph.edges(data=True))