import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from simpy import Environment
import random

from Bolt import Bolt
from Config import Config
from Data import Data, IdentityDataTransformer
from Edge import Edge
from Machine import Machine
from Spout import Spout
from Sampler import BetaSampler, PoissonSampler, IdentitySampler


class Topology():
    """
    A class that contains the executor connectivity and their assignment information
    """

    def __init__(self,
                 n_machines: int,
                 executors_info: dict,
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
        self.edge_list = []

        # key: name of the executor, value: [type of executor (in str), number of replicas]
        self.executor_info = executors_info
        # we are assuming the inter trans are using seperate bandwidth
        self.inter_trans_delay = inter_trans_delay

        self.executor_to_machines = {}
        self.machine_to_executors = {}
        self.name_to_executors = {}
        self.executor_groups = []
        self.executor_flat = []

        # tracking related
        self.tracking = False
        self.tracking_counter = 0
        self.tracking_list = []
        self.collection_counter = 0

        # this is for calculating the overall throughput
        self.total_income = 0
        self.total_finish = 0

        self.env = Environment()

        self.assignment_cache = None

        # setup random seed
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        
        print('Building Topology')
        for k, v in executors_info.items():
            if k != 'graph':
                print(k, v[1])
        

    def update_assignments(self, new_assignments, assignemt_type='condensed'):
        # self.assignment_cache = new_assignments
        self.reset_assignments()

        ##############################################################
        # NOTICE: Assuming the condensed new_assignment will take following form
        # X = [[0.4, 0.3, 0.4]] shape=(1,3)
        # first dimension represent bolt, second dimension represent
        # how many percentage on each machine, this example represent
        # one type of bolt on three machines
        ##############################################################

        if assignemt_type == 'condensed':
            assert(len(new_assignments) == len(self.executor_groups))
            assert(len(new_assignments[0]) == len(self.machine_list))
            self.assignment_cache = []
            for i in range(len(new_assignments)):
                # new assignment example: [0.4 0.3 0.4]
                num_vec = self.assignment_calibration(len(self.executor_groups[i]), new_assignments[i])
                prefix = 0
                self.assignment_cache.append(num_vec)
                if Config.debug or Config.reset:
                    print(self.executor_groups[i], num_vec)

                for j in range(len(num_vec)):
                    if num_vec[j] == 0:
                        continue
                    # print(num_vec[j])
                    suffix = prefix + num_vec[j]
                    executors = self.executor_groups[i][prefix:suffix]

                    # this is only for debug, this shouldn't be printed at all
                    if executors == []:
                        print(prefix, suffix)
                        print(num_vec)

                    # print(executors)
                    self.add_executor_to_machines(executors, self.machine_list[j])
                    self.add_machine_to_executors(executors, self.machine_list[j])
                    prefix += num_vec[j]
                
                if Config.debug or Config.reset:
                    print(self.machine_to_executors)

        elif assignemt_type == 'one-hot':
            assert(len(new_assignments) == len(self.executor_flat))
            assert(len(new_assignments[0]) == len(self.machine_list))

            for i in range(len(new_assignments)):
                index = np.argmax(new_assignments[i])
                self.add_executor_to_machines(self.executor_flat[i], self.machine_list[index])
                self.add_machine_to_executors(self.executor_flat[i], self.machine_list[index])
        else:
            raise ValueError(f'Unknown assignemt type {assignemt_type}')

    def assignment_calibration(self, machine_size, prop_vec):
        num_vec = list(map(lambda x:round(x*machine_size), prop_vec))
        # sort the prop_vec based on their weight
        # in (num_vec idx, weight)
        prop_order_list = list(sorted(enumerate(prop_vec), key=lambda x:x[1]))
        diff = sum(num_vec) - machine_size
        
        if diff == 0:
            pass
        elif diff > 0:
            # we need to assign less from the slot with less weight
            idx = 0
            while diff > 0:
                vec_idx = prop_order_list[idx][0]
                if num_vec[vec_idx] < 1:
                    idx += 1
                    continue
                num_vec[vec_idx] -= 1
                idx += 1
                diff -= 1
        else:
            # we need to assign more from the slot with more weight
            idx = -1
            while diff < 0:
                vec_idx = prop_order_list[idx][0]
                num_vec[vec_idx] += 1
                idx -= 1
                diff += 1
        
        return num_vec

    def update_states(self, time:int=100, track=False):
        # the time should represent the time interval that we would like to sample data from
        if track:
            # reset everything before we start tracking
            self.tracking = True
            self.tracking_counter = 0
            self.collection_counter = 0
            self.tracking_list = []
            self.total_income = 0
            self.total_finish = 0
            starting_time = self.env.now
            metrics = {}

            next_batch = int(round(self.env.now, 0)) + time
            self.env.run(until=next_batch)

            if Config.progress_check or Config.debug:
                print(f'In total {self.tracking_counter} tracked data')

            # turn off tracking as we already generate enough data for tracking
            self.tracking = False
            b_count = 0
            # we will only update the system for at most 10*time to prevent extreme situation
            while b_count != 2:
                if Config.progress_check or Config.debug:
                    print(
                        f'{self.collection_counter*100/self.tracking_counter:.2f} collected {b_count}')
                # perform early breaking if we received all the tracking data
                if self.collection_counter == self.tracking_counter:
                    break
                next_batch = int(round(self.env.now, 0)) + time*5
                self.env.run(until=next_batch)
                b_count += 1

            # calculate the latency
            total_delay = 0
            for d in self.tracking_list:
                if d.finish_time is not None:
                    total_delay += d.finish_time - d.enter_time

            latency = -(total_delay / self.tracking_counter)

            # if we did not get all the tracking task, simply add whole trajectary 
            # with proportion to the data we did not received as penalty
            if self.collection_counter < self.tracking_counter:
                offset = -(time*5*b_count)*(1 - (self.collection_counter/self.tracking_counter))
                if Config.debug or Config.progress_check:
                    print(f'{self.collection_counter/self.tracking_counter} collected with offset={offset}')
                latency += offset

            if Config.progress_check or Config.debug:
                print(f'final latency is {latency}')
                print(f'simulation end at {self.env.now}')

            metrics['throughput'] = self.total_finish
            metrics['unit_throughput'] = self.total_finish / (self.env.now-starting_time)
            metrics['relative_throughput'] = self.total_finish/self.total_income
            metrics['latency'] = latency

            avg_incoming_rate = []
            for sp in self.name_to_executors['spout']:
                avg_incoming_rate.append(np.mean(sp.rate_history))

            metrics['avg_incoming_rate'] = avg_incoming_rate

            return metrics
        else:
            # This should only use for debug or data collection for cold start
            next_batch = int(round(self.env.now, 0)) + time
            self.env.run(until=next_batch)

    def round_robin_init(self, shuffle=False) -> None:
        """
        Initialise the resources in a round-robin manner
        """
        assert(self.machine_list is not None)
        assert(self.machine_list != [])

        exec = []

        for e_list in self.name_to_executors.values():
            exec += e_list
        
        if shuffle:
            # this shuffle is inplaced
            random.shuffle(exec)

        r = len(self.machine_list)

        for i in range(len(exec)):
            to = i % r
            self.add_executor_to_machines(exec[i], self.machine_list[to])
            self.add_machine_to_executors(exec[i], self.machine_list[to])

    def get_downstreams(self, source) -> list:
        """
        get a list of downstream bolts of current source
        """
        try:
            if type(source) is str:
                # print(source, self.executor_graph.successors(source))
                successors = list(self.executor_graph.successors(source))
            else:
                # print(source, list(self.executor_graph.successors(source.name)))
                successors = list(
                    self.executor_graph.successors(source.name))
        except IndexError:
            # this indicate we reached the end bolt
            return []

        res = []
        for succ in successors:
            res.append(self.name_to_executors[succ])

        # print(res)
        return res

    def get_network(self, source, target) -> Edge:
        sm = self.executor_to_machines[source]
        dm = self.executor_to_machines[target]
        return self.machine_graph[sm][dm]['object']

    def record(self, data: Data) -> None:
        # self.tracking_list.append(data)
        self.collection_counter += 1
        if self.collection_counter > self.tracking_counter:
            raise ValueError('There are more tracking instance accepted than generated')

    def build_machine_graph(self, edges):
        self.machine_graph.add_nodes_from(self.machine_list)

        for s, d, w, b in edges:
            ns = self.machine_list[s]
            nd = self.machine_list[d]
            self.machine_graph.add_edge(ns, nd)

            # initialise the edge object
            ob = Edge(self.env, b)
            ob.bandwidth = w
            ob.between = [ns, nd]
            self.machine_graph[ns][nd]['weight'] = w
            self.machine_graph[ns][nd]['object'] = ob
            self.edge_list.append(ob)

        # create self-loop for communication within the machine
        for m in self.machine_list:
            self.machine_graph.add_edge(m, m)
            ob = Edge(self.env)
            ob.bandwidth = self.inter_trans_delay
            ob.between = [m, m]

            self.machine_graph[m][m]['weight'] = self.inter_trans_delay
            self.machine_graph[m][m]['object'] = ob
            self.edge_list.append(ob)

    def add_executor_to_machines(self, executor, machine):
        if type(executor) is list:
            assert(executor != [])
            for e in executor:
                self.add_executor_to_machines(e, machine)
        else:
            self.executor_to_machines[executor] = machine

    def add_machine_to_executors(self, executor, machine):
        if type(executor) is list:
            assert(executor != [])
            self.machine_to_executors[machine] = self.machine_to_executors.get(
                machine, []) + executor
        else:
            self.machine_to_executors[machine] = self.machine_to_executors.get(
                machine, []) + [executor]

    def create_spouts(self, n, param):
        assert(len(param) == n)
        for i in range(n):
            # new_spout = Spout(i, data_rates[i],
            #                   self.env, self, self.random_seed)
            new_spout = Spout(i, self.env, self, **(param[i]))
            self.name_to_executors['spout'] = self.name_to_executors.get(
                'spout', []) + [new_spout]

    def create_bolts(self, n, name, **bolt_info):
        for i in range(n):
            new_bolt = Bolt(name, i, self.env, self, **bolt_info)
            self.name_to_executors[name] = self.name_to_executors.get(
                name, []) + [new_bolt]

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
        # use this list to unpack the condensed new assignment
        self.executor_groups = list(self.name_to_executors.values())
        # a flatten version to unpack the one-hot assignment
        self.executor_flat = [i for e in self.executor_groups for i in e]

    def build_homo_machines(self, capacity=1):
        """
        Parameters
        -----------
        capacity
            build n machines all with the same computational capactiy
        """
        self.build_heter_machines(capacity_list=[capacity]*self.n_machines)

    def build_heter_machines(self, capacity_list: list):
        """
        Parameters
        -----------
        capacity_list
            build n machines with the the corresponding computational capactiy
            specified in this list
        """
        assert(len(capacity_list) == self.n_machines)
        for i in range(self.n_machines):
            m = Machine(i, self, self.env, capacity=capacity_list[0])
            self.machine_list.append(m)

    def build_sample(self):
        self._build_sample_executors()
        self._build_sample_machines()
        self.round_robin_init()
        if Config.debug:
            print(self.name_to_executors)
    
    def build_sample_complex(self):
        self._build_sample_executors_complex()
        self._build_sample_machines()
        self.round_robin_init()
        if Config.debug:
            print(self.name_to_executors)

    def _build_sample_machines(self):
        self.n_machines = 4
        self.build_homo_machines()
        edges = [(0, 1, 1e3, 100), (0, 2, 1e3, 100), (0, 3, 1e3, 100),
                 (1, 2, 1e3, 100), (1, 3, 1e3, 100), (2, 3, 1e3, 100)]
        self.build_machine_graph(edges)

    def _build_sample_executors(self):
        sample_info = {
            'spout': ['spout', 2, [
                {"rate_sampler":PoissonSampler(mu=20, random_seed=self.random_seed+offset), 
                "batch":20, 
                "random_seed":self.random_seed+offset} for offset in range(2)]
            ],
            'SplitSentence': ['bolt', 5, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':20,
                    'random_seed':None,
                }],
            'WordCount': ['bolt', 5, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':20,
                    'random_seed':None,
                }],
            'Database': ['bolt', 5, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':20,
                    'random_seed':None,
                }],
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

    def _build_sample_executors_complex(self):
        sample_info = {
            'spout': ['spout', 2, [
                {"rate_sampler":PoissonSampler(mu=20, random_seed=self.random_seed+offset), 
                "batch":20, 
                "random_seed":self.random_seed+offset} for offset in range(2)]
            ],
            'A': ['bolt', 5, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':20,
                    'random_seed':None,
                }],
            'B': ['bolt', 5, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':20,
                    'random_seed':None,
                }],
            'C': ['bolt', 5, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':20,
                    'random_seed':None,
                }],
            'D': ['bolt', 5, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':20,
                    'random_seed':None,
                }],
            'E': ['bolt', 5, {
                    'd_transform': IdentityDataTransformer(),
                    'batch':20,
                    'random_seed':None,
                }],
            'graph': [
                # we first define a list of nodes
                ['spout', 'A', 'B', 'C', 'D', 'E'],
                # then, we have edge represent in tuples
                ('spout', 'A'),
                ('A', 'B'),
                ('B', 'D'),
                ('A', 'C'),
                ('C', 'E'),
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
        nx.draw_networkx_edge_labels(
            self.machine_graph, pos, edge_labels=labels)
        plt.show()

    def draw_executors(self):
        pos = nx.spring_layout(self.executor_graph)
        nx.draw_networkx(self.executor_graph, pos)
        plt.show()

    def reset_assignments(self):
        self.machine_to_executors = {}
        self.executor_to_machines = {}

        for executors in self.name_to_executors.values():
            for e in executors:
                e.clear()
        
        for e in self.edge_list:
            e.clear()

        if Config.debug:
            print('The current config has been reset')


if __name__ == '__main__':
    # test = Topology(4, {})
    test = Topology(4, {}, random_seed=100)
    # test.build_sample()
    test.build_sample_complex()
    

    """
    Graph example
    """
    # for i in test.machine_list:
    #     print(i.capacity)
    # test.draw_machines()
    test.draw_executors()

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

    """
    Test new assignment updates
    """
    # test.update_states(time=10, track=True)
    # test.update_states(time=0.1, track=False)
    # test.update_assignments([[0.3, 0.3, 0.2, 0.2]]*6)
    # test.update_states(time=0.105, track=False)

    """
    Tracking info
    """
    # print(len(test.tracking_list))
    # print(test.tracking_counter)
    # test.update_states(time=1000, track=False)
    # test.update_states(time=1000, track=False)
    # test.update_states(time=1000, track=True)
    # test.update_states(time=1.2, track=False)
    # int_vec = test.assignment_calibration(11, [0.3, 0.2, 0.1, 0.4])
    # print(int_vec, sum(int_vec))

    """
    Test one-hot unpacking
    """
    # n_executors = len(test.executor_flat)
    # n_machines = len(test.machine_list)
    # value = np.random.randn(n_executors, n_machines)
    # col = np.argmax(value, axis=1)
    # row = np.array(range(n_executors))
    # new_ass = np.zeros(value.shape)
    # new_ass[row, col] = 1
    # print(test.machine_list)
    # print(test.executor_flat)
    # print(col)
    # print(test.machine_to_executors)
    # test.update_assignments(new_ass, assignemt_type='one-hot')
    # print('--------------------------')
    # print(test.machine_to_executors)