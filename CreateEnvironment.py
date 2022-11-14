import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random
import numpy
import pandas as pd

class Node:
    def __init__(self, i) -> None:
        self.degree = 0
        self.neighbor = []
        self.occupied_by = None
        self.isroot = False
        self.node_value = i 
        self.degree_dict = {} #Update later


class Graph_util(Node):
    def __init__(self, node_count) -> None:
        self.G = nx.cycle_graph(node_count)
        self.node_count = node_count
        self.root = None
        self.prey = None
        self.predator = None
        self.agent = None
        self.degree_dict = {2:[i for i in range(node_count)]}
        # Graph inititializing
        for i in range(node_count):
            node = Node(i)
            node.degree = 2 
            if i ==0:
                node.isroot = True
                self.root = node
                self.G.add_node( i, data = node)
            else:
                self.G.add_node( i, data = node)
        self.generate_edges()

    def get_adj_list(self):
        return self.G.adj

    def display(self):
        nx.draw_circular(self.G,
                 node_color='y',
                 node_size=150,
                 with_labels=True)
        plt.show()
    
    def display_shortest_path(self, source, s_path):
        pos = nx.circular_layout(self.G)
        nx.draw_networkx_edges(self.G, pos, edgelist=s_path,
                       width=8, alpha=0.5, edge_color='r')
        labels = {}
        for i in range(1, self.node_count):
            if i == source:
                labels[i] = r'$%d*$' % i
            else:
                labels[i] = r'$%d$' % i
        nx.draw_circular(self.G,
                 node_color='y',
                 node_size=150,
                 with_labels=True)
        nx.draw_networkx_labels(self.G, pos, labels, font_size=16)
        plt.axis('off')
        plt.show()
    
    def generate_cyclic_edges(self):
        vertices = [ i for i in range(1,self.node_count+1)]
        edge_data = [i for i in range(2, self.node_count+2)]     
        edges = [(u,edge_data[vertices.index(u)]) for u in vertices ]   
        self.G.add_edges_from(edges)
        self.G.add_edge(0,self.node_count)
    
    def get_degrees(self):
        return self.G.degree()

    def generate_edges(self):
        adj_list = self.get_adj_list()
        edges = []
        count = 0
        random_node = random.choice(self.degree_dict[2])
        for i in range(random_node, random_node+self.node_count):
            #print(i)
            if (i%self.node_count) in self.degree_dict[2] and ((i+5)%self.node_count) in self.degree_dict[2]:
                edges.append((i%self.node_count,(i+5)%self.node_count))
                self.degree_dict[2].remove(i%self.node_count)
                self.degree_dict[2].remove((i+5)%self.node_count )
                count+=1
                self.G.nodes[i%self.node_count]["data"].degree+=1
                self.G.nodes[(i+5)%self.node_count]["data"].degree+=1
        self.G.add_edges_from(edges)
        #self.display()

    def generate_egdes_from_path(self, path):
        e = []
        for i in range (len(path)-1):
            a,b = i, i+1
            e.append((a,b))
        return e

    def shortest_path(self, source, target):
        s_path = nx.shortest_path(self.G, source=self.G.nodes[source]["data"].node_value, target=self.G.nodes[target]["data"].node_value)
        #shortest_edge_path = nx.all_simple_edge_paths(graph.G, source=graph.G.nodes[source]["data"].node_value, target=graph.G.nodes[target]["data"].node_value)
        s_path_edges = self.generate_egdes_from_path(s_path) 
        return s_path, s_path_edges

    def all_shortest_paths(self):
        return nx.floyd_warshall(self.G, weight='weight')
# Simulate from multinomial distribution
def simulate_multinomial(vmultinomial):
        r=np.random.uniform(0.0, 1.0)
        CS=np.cumsum(vmultinomial)
        CS=np.insert(CS,0,0)
        m=(np.where(CS<r))[0]
        nextState=m[len(m)-1]
        return nextState

import numpy as np


def stationary_probablity(start_state):
    for i in range(50):
        steady_state = np.dot(start_state,transition_matrix)
    return steady_state
   
if __name__ == "__main__":
    graph = Graph_util(10)
    adjaceny_matrix = nx.to_numpy_matrix(graph.G)
    #adjaceny_matrix = (numpy.asanyarray(adjaceny_matrix))
    #adjaceny_matrix[0] = 1
    '''
    
    
    '''
    for i in range(len(adjaceny_matrix)):
        adjaceny_matrix[i,i] = 1
    #print(adjaceny_matrix)
    
    #input()
    transition_matrix = (adjaceny_matrix/adjaceny_matrix.sum(axis=0))
    print(transition_matrix)

    # Normalize the matrix so that rows sum to 1
    #P = adjaceny_matrix/np.sum(adjaceny_matrix, 1)[:, np.newaxis]
    #print(P)
    #start_state = np.array([ 0,0,0,0,0,0,0,0,0,0])
    
    start_state = np.array([0 for i in range(10)])
    start_state[3] = 1
    start_state = np.array([1/9 for i in range(10)])
    start_state[3] = 0
    start_state = np.array([ 0.06611570247933884,
                0.07644628099173555,
                0.10743801652892562,
                0.12603305785123967,
                0.11157024793388431,
                0.0847107438016529,
                0.09917355371900828,
                0.11983471074380166,
                0.11776859504132232,
                0.09090909090909091])
    print(start_state)
    input()
    #start_state = np.dot(start_state,transition_matrix)
    #print(start_state)
    #start_state = np.dot(start_state,transition_matrix)
    #rint(start_state)
    #exit()
    for i in range(50):
            #transition_matrix = np.dot(transition_matrix.T,transition_matrix)
            #transition_matrix = np.dot(transition_matrix.T,transition_matrix)
            start_state = np.dot(start_state,transition_matrix)
            
            print(start_state)
            input()
            
    print(start_state)
    print(start_state.sum())
    
    

