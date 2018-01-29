class Vertex(object):
    # Keep a (global) counter to 
    # give each node a unique id
    vertex_counter = 0
    
    def __init__(self, name):
        self.reachable = []
        self.name = name
        self.id = Vertex.vertex_counter
        self.previous_name = None
        Vertex.vertex_counter += 1
      

class Graph(object):
    def __init__(self):
        self.vertices = []
        # Transition matrix is a dict of dict, we can
        # access all possible transitions from a vertex
        # by indexing the transition matrix first and then
        # check whether the destination in the dict is True
        self.transition_matrix = {}
        
    def add_vertex(self, vertex):
        """Add vertex to graph and update the 
        transition matrix accordingly"""
        transition_row = {}
        for v in self.vertices:
            transition_row[v] = 0
            self.transition_matrix[v][vertex] = 0
        self.transition_matrix[vertex] = transition_row
        self.vertices.append(vertex)

    def add_edge(self, v1, v2):
        self.transition_matrix[v1][v2] = 1
        
    def remove_edge(self, v1, v2):
        self.transition_matrix[v1][v2] = 0

    def get_neighbors(self, vertex):
        nodes = self.transition_matrix[vertex].items()
        return [k for (k, v) in nodes if v == 1]

    def relabel_nodes(self, mapping):
        for v in self.vertices:
            if v in mapping:
                v.previous_name = v.name
                v.name = mapping[v]