import sys


def find_shortest_paths(matrix, start):
    """Dijkstra algorithm for O(n^2)."""
    inf = sys.maxsize
    used = [False] * len(matrix)
    distances = [inf] * len(matrix)
    number_of_paths = [1] * len(matrix)
    prev = [start] * len(matrix)
    distances[start] = 0

    for i in range(len(matrix)):
        cur_vert_dist = inf
        cur_vert_pos = 0
        for pos, distance in enumerate(distances):
            if not used[pos] and distance < cur_vert_dist:
                cur_vert_dist = distance
                cur_vert_pos = pos

        used[cur_vert_pos] = True
        curvertedges = matrix[cur_vert_pos]

        for pos, distance in enumerate(curvertedges):
            if distances[pos] > distance + cur_vert_dist:
                distances[pos] = distance + cur_vert_dist
                prev[pos] = cur_vert_pos
                number_of_paths[pos] = number_of_paths[cur_vert_pos]
            elif cur_vert_pos != pos and distances[pos] == distance + cur_vert_dist:
                number_of_paths[pos] += number_of_paths[cur_vert_pos]
    return distances, number_of_paths, prev


def find_path(start_vert, cur_vert, prev_vector):
    path = []
    while cur_vert != start_vert:
        print(cur_vert, end=' ')
        path.append(str(cur_vert))
        cur_vert = prev_vector[cur_vert]
    print(cur_vert, end=' ')
    path.append(str(start_vert))
    #return '|'.join(path[::-1])
    return path[::-1]

def get_closeness_centrality(matrix, dist):
    cc = []
    for i in range(len(matrix[0])):
        counter = 0
        for j in range(len(matrix[0])):
            if matrix[i][j] < dist:
                counter += matrix[i][j]
        cc.append(counter)
    return cc


def get_graph_centrality(matrix, dist):
    gc = []
    for i in range(len(matrix[0])):
        localmax = 0
        for j in range(len(matrix[0])):
            if dist > matrix[i][j] > localmax:
                localmax = matrix[i][j]
        gc.append(localmax)
    return gc


def get_nodes_betweenness_centrality(dists, s_dists, paths, maxdist):
    bc = [0.0] * len(dists)
    for v in range(len(s_dists)):
        for i, (d_v, s) in enumerate(s_dists[v]):
            if d_v >= maxdist: break
            for j in range(i + 1, len(s_dists)):
                d_t, t = s_dists[v][j]
                if d_v + d_t >= maxdist: break
                if v != s and d_v + d_t == dists[s][t]:
                    bc[v] += paths[s][v] * paths[v][t] / paths[s][t]
    bc = [int(round(i)) for i in bc]
    return bc