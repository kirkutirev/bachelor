from ctypes import cdll, c_int, c_void_p, py_object


class MyGraph:
    lib = cdll.LoadLibrary('./my_graph.so')

    __new_g = lib.new_graph
    __new_g.argtypes = [py_object]
    __new_g.restype = c_void_p

    __get_cm = lib.get_current_matrix
    __get_cm.argtypes = [c_void_p]
    __get_cm.restype = py_object

    __get_mm = lib.get_mindist_matrix
    __get_mm.argtypes = [c_void_p]
    __get_mm.restype = py_object

    __get_nbc = lib.get_node_betweenness_centrality
    __get_nbc.argtypes = [c_void_p, c_int]
    __get_nbc.restype = py_object

    __get_ebc = lib.get_edge_betweenness_centrality
    __get_ebc.argtypes = [c_void_p]
    __get_ebc.restype = py_object

    __reb_cur_matrix = lib.rebuild_cur_edges
    __reb_cur_matrix.argtypes = [c_void_p, c_int]
    __reb_cur_matrix.restype = None

    __find_sh_pths = lib.find_shortest_paths
    __find_sh_pths.argtypes = [c_void_p]
    __find_sh_pths.restype = None

    __dbscan = lib.dbscan
    __dbscan.argtypes = [c_void_p, c_int, c_int, c_int]
    __dbscan.restype = None

    __get_comms = lib.get_communities
    __get_comms.argtypes = [c_void_p]
    __get_comms.restype = py_object

    __print_comms = lib.print_communities
    __print_comms.argtypes = [c_void_p]
    __print_comms.restype = None

    def __init__(self, matrix):
        self._graph = MyGraph.__new_g(matrix)

    def __repr__(self):
        return 'not implemented yet'

    def get_node_betweenness_centrality(self, max_dist):
        return MyGraph.__get_nbc(self._graph, max_dist)

    def get_mindist_matrix(self):
        return MyGraph.__get_mm(self._graph)

    def get_current_matrix(self):
        return MyGraph.__get_cm(self._graph)

    def get_edge_betweenness_centrality(self):
        return MyGraph.__get_ebc(self._graph)

    def rebuild_matrix(self, max_dist):
        MyGraph.__reb_cur_matrix(self._graph, max_dist)

    def find_shortest_paths(self):
        MyGraph.__find_sh_pths(self._graph)

    def dbscan(self, max_dist, eps, m):
        MyGraph.__dbscan(self._graph, max_dist, eps, m)

    def get_communities(self):
        return MyGraph.__get_comms(self._graph)

    def print_communities(self):
        MyGraph.__print_comms(self._graph)
