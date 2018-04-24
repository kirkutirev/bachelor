from collections import OrderedDict
import math

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


def get_jaro_distance(first, second, *args, winkler=True, winkler_ajustment=True, scaling=0.1):
    first = ''.join(first[0])
    second = ''.join(second[0])
    jaro = score(first, second)
    cl = min(len(get_prefix(first, second)), 4)

    if all([winkler, winkler_ajustment]):
        return int(round((1 - (jaro + (scaling * cl * (1.0 - jaro)))) * 100))

    return int(round((1 - jaro) * 100))


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
