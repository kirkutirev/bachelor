import matplotlib.pyplot as plt
import networkx as nx
import re
import operator
import math
import sys
from my_metrics import *
from my_graph_functions import *
from my_i_o_functions import *
from copy import deepcopy
from collections import OrderedDict
from pandas import Series
from pylab import ceil, title, xlabel, ylabel
from statistics import median


def merge_distances(d):
    newd = []
    for curmassd in range(1, len(d)):
        for curd in range(curmassd + 1, len(d)):
            newd.append(d[curmassd][curd])
    return newd


def get_adj_matrix(clusters, dist_func, deps_labels):
    matrix = [[0] * len(clusters) for i in range(len(clusters))]
    for fst_pos, fst_pers in enumerate(clusters):
        for snd_pos, snd_pers in enumerate(clusters):
            matrix[fst_pos][snd_pos] = dist_func(fst_pers, snd_pers, deps_labels)
    return matrix


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
