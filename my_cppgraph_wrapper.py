from ctypes import cdll, c_int, c_void_p, py_object


class MyGraph:
    lib = cdll.LoadLibrary('./my_graph.so')

    _new_g = lib.new_graph
    _new_g.argtypes = [py_object]
    _new_g.restype = c_void_p

    _get_cm = lib.get_current_matrix
    _get_cm.argtypes = [c_void_p]
    _get_cm.restype = py_object

    _get_mm = lib.get_mindist_matrix
    _get_mm.argtypes = [c_void_p]
    _get_mm.restype = py_object

    _get_nbc = lib.get_node_betweenness_centrality
    _get_nbc.argtypes = [c_void_p, c_int]
    _get_nbc.restype = py_object

    def __init__(self, matrix):
        self._graph = MyGraph._new_g(matrix)

    def __repr__(self):
        return 'not implemented yet'

    def get_node_betweenness_centrality(self, maxdist):
        return MyGraph._get_nbc(self._graph, maxdist)

    def get_mindist_matrix(self):
        return MyGraph._get_mm(self._graph)


if __name__ == '__main__':
    print('ok')
