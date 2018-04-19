import matplotlib.pyplot as plt
import networkx as nx
import re
import operator
from copy import deepcopy
from collections import OrderedDict
from pandas import Series
from pylab import *
from statistics import median

'''
def read_file(filename):
    file = open(filename, 'r')
    clusters = []
    data = file.readlines()
    counter = -1
    for curString in data:
        if re.match(r'Cluster \d \d+', curString):
            clustersinfo = [int(curNumber) for curNumber in re.findall(r'\d+', curString)]
            print(clustersinfo)
            clusters.append([])
            counter += 1
        else:
            place = re.findall(r'\'(\w)\'', curString)
            time = [int(curTime) for curTime in re.findall(r' (\d+)', curString)]
            clusters[counter].append((place, time))
    file.close()
    return clusters

def print_clusters(clusters):
    for curcluster in clusters:
        print(curcluster)
        for curperson in curcluster:
            print(curperson[0], curperson[1])
    pass
'''


def read_file(filename):
    file = open(filename, 'r')
    clusters = []
    data = file.readlines()
    counter = -1
    for curString in data:
        if re.match(r'Cluster \d \d+', curString):
            clustersinfo = [int(curNumber) for curNumber in re.findall(r'\d+', curString)]
            #print(clustersinfo)
            counter += 1
        else:
            place = re.findall(r'\'(\w)\'', curString)
            time = [int(curTime) for curTime in re.findall(r' (\d+)', curString)]
            clusters.append((place, time, counter))
    file.close()
    return clusters


def read_matrix_from_file(filename):
    file = open(filename, 'r')
    temp = [int(i) for i in file.read().split()]
    pointer = 1
    matrix = [[0] * temp[0] for i in range(temp[0])]
    for i in range(temp[0]):
        for j in range(temp[0]):
            matrix[i][j] = temp[pointer]
            pointer += 1
    file.close()
    return matrix


def read_bc_from_file(filename):
    file = open(filename, 'r')
    data = [int(i) for i in file.read().split()]
    data = [i[1] for i in enumerate(data) if i[0] > 0]
    file.close()
    return data


def read_edges_from_file(filename):
    file = open(filename, 'r')
    data = []
    for line in file.readlines():
        cur_line = [int(i) for i in line.split()]
        data.append((cur_line[0], cur_line[1]))
    file.close()
    return data


def print_matrix_into_file(filename, matrix):
    out = open(filename, 'w')
    out.write("{0}\n".format(len(matrix[0])))
    for i in range(len(matrix[0])):
        for j in range(len(matrix[0])):
            out.write("{0} ".format(str(matrix[i][j])))
        out.write('\n')
    out.close()
    pass


def print_clusters_into_run(clusters):
    for curperson in clusters:
        print(curperson[0], curperson[1], curperson[2])
    pass


def get_levenshtein_distance(first, second, *args):
    fst_places = ''.join(first[0])
    snd_places = ''.join(second[0])
    fst_times = first[1]
    snd_times = second[1]
    insCost = 50
    delCost = 50
    repCost = 100
    d = [[0] * (len(snd_places) + 1) for i in range(len(fst_places) + 1)]
    for j in range(1, len(snd_places) + 1):
        d[0][j] = d[0][j - 1] + insCost
    for i in range(1, len(fst_places) + 1):
        d[i][0] = d[i - 1][0] + delCost
        for j in range(1, len(snd_places) + 1):
            if fst_places[i - 1] != snd_places[j - 1]:
                d[i][j] = min(d[i - 1][j] + delCost, d[i][j - 1] + insCost, d[i - 1][j - 1] + repCost)
            else:
                d[i][j] = d[i - 1][j - 1] + abs(fst_times[i - 1] - snd_times[j - 1])
    return d[len(fst_places)][len(snd_places)]


def merge_distances(d):
    newd = []
    for curmassd in range(1, len(d)):
        for curd in range(curmassd + 1, len(d)):
            newd.append(d[curmassd][curd])
    return newd


def get_jaro_distance(first, second, *args, winkler=True, winkler_ajustment=True, scaling=0.1):
    first = ''.join(first[0])
    second = ''.join(second[0])
    jaro = score(first, second)
    cl = min(len(get_prefix(first, second)), 4)

    if all([winkler, winkler_ajustment]):
        return int(round((1 - (jaro + (scaling * cl * (1.0 - jaro)))) * 100.0))

    return int(round(((1 - jaro) * 100)))


def score(first, second):
    shorter, longer = first.lower(), second.lower()

    if len(first) > len(second):
        longer, shorter = shorter, longer

    m1 = get_matching_characters(shorter, longer)
    m2 = get_matching_characters(longer, shorter)

    if len(m1) == 0 or len(m2) == 0:
        return 0.0

    return (float(len(m1)) / len(shorter) +
            float(len(m2)) / len(longer) +
            float(len(m1) - transpositions(m1, m2)) / len(m1)) / 3.0


def get_diff_index(first, second):
    if first == second:
        return -1

    if not first or not second:
        return 0

    max_len = min(len(first), len(second))
    for i in range(0, max_len):
        if not first[i] == second[i]:
            return i

    return max_len


def get_prefix(first, second):
    if not first or not second:
        return ""

    index = get_diff_index(first, second)
    if index == -1:
        return first

    elif index == 0:
        return ""

    else:
        return first[0:index]


def get_matching_characters(first, second):
    common = []
    limit = math.floor(min(len(first), len(second)) / 2)

    for i, l in enumerate(first):
        left, right = int(max(0, i - limit)), int(min(i + limit + 1, len(second)))
        if l in second[left:right]:
            common.append(l)
            second = second[0:second.index(l)] + '*' + second[second.index(l) + 1:]

    return ''.join(common)


def transpositions(first, second):
    return math.floor(len([(f, s) for f, s in zip(first, second) if not f == s]) / 2.0)


def get_euclidian_distance(first, second, deps_labels):
    vect_f = object_to_vector(first, deps_labels)
    vect_s = object_to_vector(second, deps_labels)
    under_root = list(map(lambda x: (x[0] - x[1]) ** 2, zip(vect_f.values(), vect_s.values())))
    return int(round(math.sqrt(sum(under_root))))


def get_block_distance(first, second, deps_labels):
    vect_f = object_to_vector(first, deps_labels)
    vect_s = object_to_vector(second, deps_labels)
    abs_diff = list(map(lambda x: abs(x[0] - x[1]), zip(vect_f.values(), vect_s.values())))
    return sum(abs_diff)


def get_cosine_distance(first, second, deps_labels):
    vect_f = object_to_vector(first, deps_labels)
    vect_s = object_to_vector(second, deps_labels)
    fst_len = math.sqrt(sum(list(map(lambda x: x ** 2, vect_f.values()))))
    snd_len = math.sqrt(sum(list(map(lambda x: x ** 2, vect_s.values()))))
    prod = sum(list(map(lambda x: x[0] * x[1], zip(vect_f.values(), vect_s.values()))))
    return int(round((1 - prod / fst_len / snd_len) * 100))


def object_to_vector(obj, deps_labels):
    vect = OrderedDict(zip(deps_labels, [0] * len(deps_labels)))
    for place, time in zip(obj[0], obj[1]):
        vect[place] += time
    return vect


def get_adj_matrix(clusters, dist_func, deps_labels):
    matrix = [[0] * len(clusters) for i in range(len(clusters))]
    for fst_pos, fst_pers in enumerate(clusters):
        for snd_pos, snd_pers in enumerate(clusters):
            matrix[fst_pos][snd_pos] = dist_func(fst_pers, snd_pers, deps_labels)
    return matrix


def dijkstra(matrix, start):
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
        path.append(str(cur_vert))
        cur_vert = prev_vector[cur_vert]
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


def get_betweenness_centrality(dists, s_dists, paths, maxdist):
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


def draw_histograms(sets_of_points, ttl='unknown', yl='Quantity', xl='------', *args, **kwargs):
    for cur_set in sets_of_points:
        plt.hist(cur_set, bins=len(set(cur_set)), alpha=0.8)
    ylabel(yl)
    xlabel(xl)
    title(ttl)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.show()
    pass


def draw_pdf(points, ttl='unknown', xl='------'):
    ser = Series(points)
    ser.plot(kind='kde', linewidth=1)
    xlabel(xl)
    title(ttl)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.show()
    pass


def draw_graph(g, node_colors, node_labels, edge_labels):
    positions = nx.spring_layout(g)
    print('graph drawing')
    nx.draw_networkx_nodes(g, pos=positions, node_color=node_colors, node_size=20)
    nx.draw_networkx_edges(g, pos=positions, width=0.05, alpha=0.2)
    nx.draw_networkx_labels(g, pos=positions, labels=node_labels, font_size=5)
    #nx.draw_networkx_edge_labels(g, pos=positions, labels=edge_labels, font_size=5)
    plt.axis('off')
    # nx.draw(g, with_labels=False, node_color=colors, node_size=5, style='dashed', width=0.05)
    # nx.draw_networkx_labels(g, pos=nx.spring_layout(g), labels=labels, font_size=5)

    # txt = 'cur d = ' + str(cur_d)
    # for i in range(len(node_colors)):
    #    txt += '\ncluster ' + str(i) + ': ' + node_colors[i]
    # plt.text(0.5, 0.5,
    #         txt,
    #         fontsize=10,
    #         ha='left',
    #        bbox=dict(alpha=0.1))

    plt.show()
    pass


def normalization(obj, c_mins, c_maxs, n_min, n_max):
    n_obj = deepcopy(obj)
    for i, dep in enumerate(obj[0]):
        n_obj[1][i] = round(n_min + (obj[1][i] - c_mins[dep]) * (n_max - n_min) / (c_maxs[dep] - c_mins[dep]))
    return n_obj


def normalize(data, departments, n_min=1, n_max=100):
    seprated_data = split_departments_data(data, departments)
    minimums = {i: min(seprated_data[i], key=lambda x: x[0])[0] for i in departments}
    maximums = {i: max(seprated_data[i], key=lambda x: x[0])[0] for i in departments}
    normalized = list([normalization(obj, minimums, maximums, n_min, n_max) for obj in data])
    return normalized


def split_departments_data(data, departments):
    separated_data = {i: [] for i in departments}
    for pos, obj in enumerate(data):
        for place, time in zip(obj[0], obj[1]):
            separated_data[place].append((time, pos))
    return separated_data


def get_rejection(data, department):
    # print('department {0} data: {1}'.format(department, len(data)))
    data.sort()
    q1 = median([i[0] for i in data][:int(ceil(len(data) / 2))])
    q3 = median([i[0] for i in data][int(ceil(len(data) / 2)):])
    # left = q1 - 1.5 * (q3 - q1)
    left = 1
    right = q3 + 1.5 * (q3 - q1)
    rejection = set(i[1] for i in data if right < i[0] or i[0] < left)
    # print('bad amount = {0}'.format(len(rejection)))
    # print('bad percent = {0}'.format(100 * len(rejection) / len(data)))
    return rejection


def remove_rejection(data, departments):
    separated_data = split_departments_data(data, departments)
    rejection = set()
    for department in separated_data:
        rejection |= get_rejection(separated_data[department], department)
    good_positions = set(range(len(data))) - rejection
    good_data = [data[i] for i in good_positions]
    return good_data


if __name__ == "__main__":
    departments = 'ADEIFN'

    print('work with clusters')
    clusters = read_file('Clusters_with_duration_KMeans7_L.txt')
    clusters = remove_rejection(clusters, departments)
    clusters = normalize(clusters, departments, n_min=1, n_max=100)

    algo = 'levenshtein'
    # algo = 'jaro'
    # algo = 'euclidian'
    # algo = 'cosine'
    # algo = 'block'

    cur_d = 100

    """
    print('matrix evaluating')
    matrix = get_adj_matrix(clusters, get_block_distance, departments)
    shortest_paths = []
    number_of_paths = []
    prev = []

    print('dijkstra evaluating')
    for i in range(len(clusters)):
        temp = dijkstra(matrix, i)
        shortest_paths.append(temp[0])
        number_of_paths.append(temp[1])
        prev.append(temp[2])

    print('printing matrices to files')
    print_matrix_into_file('dijkstra for ' + algo + ' distance.txt', shortest_paths)
    print_matrix_into_file('number of paths for ' + algo + ' distance.txt', number_of_paths)
    print_matrix_into_file('previous vertices for ' + algo + ' distance.txt', prev)
    """

    print('reading matrices from files')
    shortest_paths = read_matrix_from_file('dijkstra for ' + algo + ' distance.txt')
    number_of_paths = read_matrix_from_file('number of paths for ' + algo + ' distance.txt')
    prev = read_matrix_from_file('previous vertices for ' + algo + ' distance.txt')

    # for improving evaluating betweenness centrality
    s_shortest_paths = [sorted([(w, vert) for vert, w in enumerate(s)]) for s in shortest_paths]

    node_colors = {
        0: 'red',
        1: 'purple',
        2: 'yellow',
        3: 'blue',
        4: 'brown',
        5: 'orange',
        6: 'pink'
    }

    print('betweenness centrality evaluating')
    d = [10, 15, 20, 30, 50, 80, 100]

    if cur_d not in d:
        d.append(cur_d)
    d.sort()

    d = d[::-1]
    all_bc = []
    for i in d:
        print('betweenness centrality for d = {}'.format(i))
        all_bc.append(get_betweenness_centrality(shortest_paths, s_shortest_paths, number_of_paths, i))

    # print('drawing histograms')
    # draw_histograms(sets_of_points=all_bc, xl='Betweenness centrality', ttl=algo + ' distance')

    print('graph initializing')
    cur_d_pos = d.index(cur_d)
    edges = [(i, j) for i in range(len(shortest_paths)) for j in range(i + 1, len(shortest_paths)) if shortest_paths[i][j] < cur_d]
    edges = [(i, j) for (i, j) in edges if all_bc[cur_d_pos][i] > 0 and all_bc[cur_d_pos][j] > 0]

    g = nx.Graph()
    for i in edges:
        g.add_edge(i[0], i[1], weight=shortest_paths[i[0]][i[1]])

    node_colors = [node_colors[clusters[i][2]] for i in g.nodes()]
    node_labels = {i: str(all_bc[cur_d_pos][i]) for i in g.nodes()}
    edge_labels = {i: str(shortest_paths[i[0]][i[1]]) for i in g.edges()}

    draw_graph(g, node_colors, node_labels, edge_labels)

    print('ok')
