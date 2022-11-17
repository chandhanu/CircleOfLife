import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random
import numpy
import pandas as pd

def shortest_path(graph, distance, previous, source):
    queue = []
    
    # Initialization of distance and previous
    for node in graph.nodes():
        distance[node] = None
        previous[node] = 0
    
    distance[source] = 0
    queue.append(source)

    while len(queue) != 0:
        v = queue.pop(0)
        for w in graph.neighbors(v):
            if distance[w] == None:
                previous[w] = v
                distance[w] = distance[v] + 1
                queue.append(w)

def get_path(t, previous):
    edges = []

    while previous[t] != 0:
        edges.append((t, previous[t]))
        t = previous[t]
        
    return edges

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
        self.adjaceny_matrix = nx.to_numpy_matrix(self.G)

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
        if random.choice(["front", "back"]) == "front":
            x = random_node
            y = random_node+self.node_count
            step = 1
        else:
            y = random_node
            x = random_node+self.node_count
            step = -1
        for i in range(x, y, step):
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

if __name__ == "__main__":
    graph = Graph_util(50)
    graph.display()
    adjaceny_matrix = nx.to_numpy_matrix(graph.G)

    
    
    

