from ctypes import cdll, c_int, c_void_p, py_object


class MyGraph:
    lib = cdll.LoadLibrary('./my_graph.so')

    new_g = lib.new_graph
    new_g.argtypes = [py_object]
    new_g.restype = c_void_p

    find_gi = lib.find_graph_info
    find_gi.argtypes = [c_void_p]
    find_gi.restype = None

    get_cm = lib.get_current_matrix
    get_cm.argtypes = [c_void_p]
    get_cm.restype = py_object

    get_nbc = lib.get_node_betweenness_centrality
    get_nbc.argtypes = [c_void_p, c_int]
    get_nbc.restype = py_object

    def __init__(self, matrix):
        self.graph = MyGraph.new_g(matrix)

    def __repr__(self):
        return 'not implemented yet'

    def get_node_betweenness_centrality(self, maxdist):
        return MyGraph.get_nbc(self.graph, maxdist)

    def find_graph_info(self):
        MyGraph.find_gi(self.graph)


if __name__ == '__main__':
    print('ok')
