import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community
import matplotlib.colors as mcolors
from itertools import islice
from my_metrics import *
from my_graph_functions import *
import my_cppgraph_wrapper as mygraph
from my_i_o_functions import *
from copy import deepcopy
from pandas import Series
from pylab import ceil, title, xlabel, ylabel
from statistics import median
import graphviz as gv


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

    plt.show()
    pass


def draw_community(community, typical_vertices):
    g = gv.Graph(format='png')
    for first in range(len(community)):
        for second in range(first + 1, len(community)):
            g.edge(str(community[first]), str(community[second]))
    print('draw')
    g.render()


def normalization(obj, c_mins, c_maxs, n_min, n_max):
    n_obj = deepcopy(obj)
    for i, dep in enumerate(obj[0]):
        n_obj[1][i] = round(n_min + (obj[1][i] - c_mins[dep]) * (n_max - n_min) / (c_maxs[dep] - c_mins[dep]))
    return n_obj


def normalize(data, departments, new_min=1, new_max=100):
    seprated_data = split_departments_data(data, departments)
    minimums = {i: min(seprated_data[i], key=lambda x: x[0])[0] for i in departments}
    maximums = {i: max(seprated_data[i], key=lambda x: x[0])[0] for i in departments}
    normalized = list([normalization(obj, minimums, maximums, new_min, new_max) for obj in data])
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


def get_colors_for_clusters(number_of_clusters):
    colors = list(islice(mcolors.cnames.keys(), number_of_clusters))
    return colors


def find_typical_vertices(communities, init_matrix):
    vertices = []
    for community in communities:
        cnt = [0] * len(community)
        for i, s in enumerate(community):
            for j, t in enumerate(community):
                if i != j:
                    cnt[i] += init_matrix[s][t]
                    cnt[j] += init_matrix[t][s]
        pos = 0
        for i, value in enumerate(cnt):
            if value < cnt[pos]:
                pos = i
        vertices.append(pos)
    return vertices


if __name__ == "__main__":
    departments = 'ADEIFN'

    print('work with clusters')
    clusters = read_file('Clusters_with_duration_KMeans7_L.txt')
    clusters = remove_rejection(clusters, departments)
    clusters = normalize(clusters, departments, new_min=1, new_max=100)

    algo = 'levenshtein'
    # algo = 'jaro'
    # algo = 'euclidian'
    # algo = 'cosine'
    # algo = 'block'

    cur_d = 50

    # print('matrix evaluating')
    # matrix = get_adj_matrix(clusters, get_levenshtein_distance, departments)
    # # print_matrix_into_file('temp' + algo + '.txt', matrix=matrix)
    # print('graph building')
    # g = mygraph.MyGraph(matrix)
    # # g.find_shortest_paths()
    # print('dbscan')
    # # shortest_paths = g.get_mindist_matrix()
    # g.dbscan(100, 50, 3)
    # communities = g.get_communities()
    # for number, community in enumerate(communities):
    #     print(number, len(community), end=' ')
    # print()
    # print('drawing')

    # for community in communities:
    #     print(len(community))
    #     draw_community(community, matrix)

    """
    print('printing matrices to files')
    print_matrix_into_file('dijkstra for ' + algo + ' distance.txt', shortest_paths)
    print_matrix_into_file('number of paths for ' + algo + ' distance.txt', number_of_paths)
    print_matrix_into_file('previous vertices for ' + algo + ' distance.txt', prev)
    """
    print('ok')
