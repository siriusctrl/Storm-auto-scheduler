import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import simpy
from simpy import Environment

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
        # we are assuming the inter trans are using seperate bandwidth
        self.inter_trans_delay = inter_trans_delay

        self.executor_to_machines = {}
        self.machine_to_executors = {}
        self.name_to_executors = {}

        self.random_seed = random_seed
        self.track_counter = 0
        self.track_datas = {}

        self.env = Environment()

        # setup random seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def update_assignments(self, new_assignments):
        pass

    def update_states(self, time:int=1000, track=False):
        # TODO: use the track and debug flag
        self.env.run(until=time)

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

    def get_network(self, source, target) -> Edge:
        sm = self.executor_to_machines[source]
        dm = self.executor_to_machines[target]
        return self.machine_graph[sm][dm]['object']

    def build_machine_graph(self, edges):
        self.machine_graph.add_nodes_from(self.machine_list)

        for s, d, w in edges:
            ns = self.machine_list[s]
            nd = self.machine_list[d]
            self.machine_graph.add_edge(ns, nd)

            # initialise the edge object
            ob = Edge(self.env)
            ob.bandwidth = w
            self.machine_graph[ns][nd]['weight'] = w
            self.machine_graph[ns][nd]['object'] = ob
        
        # create self-loop for communication within the machine
        for m in self.machine_list:
            self.machine_graph.add_edge(m, m)
            ob = Edge(self.env)
            ob.bandwidth = self.inter_trans_delay
            
            self.machine_graph[m][m]['weight'] = self.inter_trans_delay
            self.machine_graph[m][m]['object'] = ob

    def add_executor_to_machines(self, executor, machine):
        self.executor_to_machines[executor] = machine
        
    def add_machine_to_executors(self, executor, machine):
        self.machine_to_executors[machine] = self.machine_to_executors.get(machine, []) + [executor]

    def create_spouts(self, n, data_rates):
        assert(len(data_rates) == n)

        for i in range(n):
            new_spout = Spout(i, data_rates[i], self.env, self, self.random_seed)
            self.name_to_executors['spout'] = self.name_to_executors.get('spout', []) + [new_spout]
  
    def create_bolts(self, n, name, **bolt_info):
        for i in range(n):
            new_bolt = Bolt(name, i, self.env, **bolt_info)
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
        self._build_sample_executors()
        self._build_sample_machines()
        self.round_robin_init()
        if debug:
            print(self.name_to_executors)

    def _build_sample_machines(self):
        self.n_machines = 4
        self.build_homo_machines()
        edges = [(0,1,20), (0,2,30), (0,3,15), (1,2,35), (1,3,25), (2,3,35)]
        self.build_machine_graph(edges)

    def _build_sample_executors(self):
        sample_info = {
            'spout':['spout', 2, [50, 50]],
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

    # test.update_states()
    # print(test.machine_graph.edges(data=True))
    test.update_states(time=1.5)