from CreateEnvironment import Graph_util, Node
import networkx as nx
import random
import numpy as np

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
    nodes = []
    while previous[t] != 0:
        edges.append((t, previous[t]))
        nodes.append(t)
        t = previous[t]
    return edges,nodes

class Prey(Graph_util):
    def __init__(self, graph,node_count):
        self.value = random.choice([i for i in range(node_count)])
        self.node = graph.G.nodes[self.value]
        graph.prey = self.value
        self.G = graph.G
        self.path = []
        self.path.append(self.value)

    
    def move(self):
        neighbors = list(self.G.neighbors(self.value))
        possible_nodes = neighbors[:]
        possible_nodes.append(self.value)
        self.value = random.choice(possible_nodes) # Choosing next node = random(curr_position + neighbor_nodes)
        self.node = self.G.nodes[self.value]
        self.G.prey = self.value
        self.path.append(self.value)
    
    def get_position(self):
        return self.value
           

class Predator(Graph_util):
    def __init__(self,graph,node_count):
        self.value = random.choice([i for i in range(node_count)])
        self.node = graph.G.nodes[self.value]
        graph.predator = self.value
        self.G = graph.G
        self.path = []
        self.path.append(self.value)

    def move(self, agent_pos):
        neighbors = list(self.G.neighbors(self.value))
        possible_nodes = neighbors[:]
        possible_nodes.append(self.value)
        all_pair_shortest_paths = dict(nx.all_pairs_shortest_path(self.G))
        d = {}
        min = np.inf
        for i in possible_nodes:
            i_path = all_pair_shortest_paths[i][agent_pos]
            l=len(i_path)
            if l<min:
                min = l 
            if l in d:
                d[l].append(i_path)
            else:
                d[l] = []
                d[l].append(i_path)
        
        road_to_kill_agent = random.choice(d[min])
        #print(road_to_kill_agent)
        self.value = road_to_kill_agent[0]
        self.node = self.G.nodes[self.value]
        self.G.predator = self.value
        self.path.append(self.value)
    
    def get_position(self):
        return self.value


if __name__ == "__main__":
    graph = Graph_util(50)
    prey = Prey(graph,graph.node_count)
    predator = Predator(graph,graph.node_count)

    print("prey", prey.node["data"].node_value)
    print("predator", predator.value)

    prey.move()
    print("prey new", prey.node["data"].node_value, graph.prey)
    agent_pos = 9
    predator.move(9)
    print("predator new", predator.node["data"].node_value, graph.predator)

    dist = {}
    prev = {}
    src = graph.predator
    target = agent_pos
    shortest_path(graph.G, dist, prev, src)
    s_path_edges, s_path = get_path(target, prev)
    print(s_path)
    graph.display()

    
