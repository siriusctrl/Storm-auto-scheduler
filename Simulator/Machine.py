from networkx.drawing.nx_pylab import draw_networkx, draw_networkx_edge_labels
from networkx.readwrite.edgelist import read_weighted_edgelist
import numpy as np


class Machine():
    """
    A class that represent a physical machines on the simulator
    """

    def __init__(self,
                id,
                max_slots
            ) -> None:
        
        self.id = id
        self.max_slots = max_slots
        # TODO: add info about CPU capacity later?
    
    def __repr__(self) -> str:
        return f'm{self.id}'

if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.Graph()
    a = Machine(0, 10)
    b = Machine(1, 10)
    c = Machine(2, 10)
    d = Machine(3, 10)
    G.add_nodes_from([a,b,c,d])
    # the communication delay between machine a and b is 2ms
    G.add_weighted_edges_from([(a, b, 2.), (a,c,3.), (a,d,1.5), (b,c,3.5), (b,d,2.5), (c,d,3.5)])

    # this layout can rescale edge based on edge weight
    pos = nx.kamada_kawai_layout(G)
    # draw edges and weights
    labels = nx.get_edge_attributes(G, 'weight')
    # draw the graph
    nx.draw(G, pos, with_labels=True)
    # draw the edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

    # to get the weight of edge
    print(G[a][b]['weight'])