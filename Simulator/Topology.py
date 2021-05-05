from networkx.exception import NetworkXError
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.testing._private.utils import raises
from Bolt import Bolt

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

        # key: name of the executor, value: [type of executor (in str), number of replicas]
        self.executor_info = executors_info

        self.executor_to_machines = {}
        self.machine_to_executors = {}
        self.name_to_executors = {}

        np.random.seed(random_seed)

    def update(self):
        """
        This method can update the interal state the executor assignments
        """
        pass

    def round_robin_init(self):
        """
        TODO: Initialise the resources in a round-robin manner
        """
        pass

    def get_next(self, source) -> list:
        """
        get a list of downstream bolts of current source
        """
        if type(source) is str:
            successor = list(self.executor_graph.successors(source))[0]
        else:
            successor = list(self.executor_graph.successors(source.name))[0]
    
        return self.name_to_executors[successor]

    def get_trans_delay(self, source, destination):
        """
        get the transmission delay between source and destination bolt
        """
        # we first retrive the host physical machine of each bolt
        source_m = self.executor_to_machines[source]
        dest_m = self.executor_to_machines[destination]
        # and then retrieve the trans delay between those two machines
        # ! this will return either the cost or 0 if both bolt on the same machine
        return self.machine_graph.get_edge_data(source_m, dest_m, default=0)

    def build_machine_graph(self, node, edges):
        self.machine_graph.add_nodes_from(node)
        self.machine_graph.add_weighted_edges_from(edges)

    def add_to_machines(self, executor, machine):
        self.executor_to_machines[executor] = machine
        self.machine_to_executors = self.machine_to_executors.get(machine, []) + [executor]

    def add_machine_to_executors(self, executor, machine):
        self.executor_to_machines[executor] = machine

    def create_spouts(self, n, data_rates):
        assert(len(data_rates) == n)

        for i in range(n):
            new_spout = Spout(i, data_rates[0])
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

    def build_sample(self, debug=False):
        self._build_sample_graph()
        self._build_sample_executors()
        if debug:
            print(self.name_to_executors)
    
    def _build_sample_graph(self):
        if self.n_machines == 4:
            a = Machine(0, 10)
            b = Machine(1, 10)
            c = Machine(2, 10)
            d = Machine(3, 10)
            edges = [(a,b,2.), (a,c,3.), (a,d,1.5), (b,c,3.5), (b,d,2.5), (c,d,3.5)]
            self.build_machine_graph([a,b,c,d], edges)
        else:
            raise ValueError('The number of machines provided does not match with the sample')

    def _build_sample_executors(self):
        sample_info = {
            'spout':['spout', 1, [1e3]],
            'SplitSentence':['bolt', 3, {}],
            'WordCount':['bolt', 3, {}],
            'Database':['bolt', 3, {}],
            'graph': [
                        ['spout', 'SplitSentence', 'WordCount', 'Database'], 
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


if __name__ == '__main__':
    test = Topology(4, {})
    test.build_sample(debug=False)

    """
    Graph example
    """
    # test.draw_machines()
    # test.draw_executors()

    """
    Name to executor example
    """
    # print(test.name_to_executors['spout'][0].name)

    """
    Get next example
    """
    # print(test.get_next('SplitSentence'))
