import re

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
    with open(filename, 'r') as file:
        data = file.readlines()
    clusters = []
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
    return clusters


def read_matrix_from_file(filename):
    with open(filename, 'r') as file:
        temp = [int(i) for i in file.read().split()]
    pointer = 1
    matrix = [[0] * temp[0] for i in range(temp[0])]
    for i in range(temp[0]):
        for j in range(temp[0]):
            matrix[i][j] = temp[pointer]
            pointer += 1
    return matrix


def read_bc_from_file(filename):
    with open(filename, 'r') as file:
        data = [int(i) for i in file.read().split()]
    data = [i[1] for i in enumerate(data) if i[0] > 0]
    return data


def read_edges_from_file(filename):
    with open(filename, 'r') as file:
        data = []
        for line in file.readlines():
            cur_line = [int(i) for i in line.split()]
            data.append((cur_line[0], cur_line[1]))
    return data


def print_matrix_into_file(filename, matrix):
    with open(filename, 'w') as out:
        out.write("{0}\n".format(len(matrix[0])))
        for i in range(len(matrix[0])):
            for j in range(len(matrix[0])):
                out.write("{0} ".format(str(matrix[i][j])))
            out.write('\n')
    pass


def print_clusters_into_run(clusters):
    for curperson in clusters:
        print(curperson[0], curperson[1], curperson[2])
    pass