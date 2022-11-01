import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import random

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
if __name__ == "__main__":
    graph = Graph_util(27)
    #graph.generate_edges()
    #print(graph.G.nodes[0]["data"].degree)
    #graph.display()

    #s_path = nx.shortest_path(graph.G, source=graph.G.nodes[0]["data"].node_value, target=graph.G.nodes[5]["data"].node_value)
    source = 0
    target = 7
    s_path, s_path_edges = graph.shortest_path(source,target)
    print("Floyd-warshall algorithm--------------")
    for k,v in graph.all_shortest_paths().items():
        print(k, v )
    print("All shortest pair length ------------")
    path = dict(nx.all_pairs_shortest_path(graph.G))
    for k,v in path.items():
        print(k, v )
    #print(    graph.G.nodes[4]["data"].node_value)    #shortest_path(graph.G, dist, prev, graph.G.nodes[0]["data"].node_value)
    #for i in (graph.G.neighbors(4)):
    #    print("neighbor", i)
    #print(graph.G.nodes())
    graph.display()
    #s_path = get_path(graph.G.nodes[5]["data"].node_value, prev)
    #print(s_path)
    #print(graph.G.adj)
    #print(graph.G.degree())
    #graph.display_shortest_path(source, s_path_edges)
