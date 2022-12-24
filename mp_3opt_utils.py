import numpy as np
import random
from scipy.spatial import distance_matrix


def create_tour(N):
    print("create tour")
    """
    Create an initial tour for the TSP
    :param int tour_length: Tour length
    :param bool rand:  Generate random tour
    :return: list with a TSP tour
    """

    tour = random.sample(range(N), N)

    return list(tour)


def calculate_distances(positions):
    """
    Calculate a all distances between poistions
    :param np.array positions: Positions of (tour_len, 2) points
    :return: list with all distances
    """

    # def length(x, y):
    #     return np.linalg.norm(np.asarray(x) - np.asarray(y))
    # distances = [[length(x, y) for y in positions] for x in positions]

    distances = distance_matrix(positions, positions)
    return distances


def route_distance(tour, distances):
    """
    Calculate a tour distance (including 0)
    :param list tour: TSP tour
    :param list : list with all distances
    :return dist: Distance of a tour
    """
    dist = 0
    prev = tour[-1]
    for node in tour:
        dist += distances[int(prev)][int(node)]
        prev = node
    return dist


# This method is used in for NN training to provide valid 3opt indices
def list_indices(n_points):
    indices_list = []
    for i in range(n_points):
        for j in range(i + 1, n_points - 1):
            for k in range(j + 1, n_points - 2 + (i > 0)):
                indices_list.append([i, j, k])
    return indices_list


def swapEdgesTwoOPT(tour, i, j):
    """
		Method to swap two edges and replace with
		their cross.
		"""
    newtour = tour[:i + 1]
    newtour.extend(reversed(tour[i + 1:j + 1]))
    newtour.extend(tour[j + 1:])

    return newtour


def swapEdgesThreeOPT(tour, i, j, k, case):
    """
		Method to swap edges from 3OPT
		"""
    if case == 1:
        newtour = swapEdgesTwoOPT(tour.copy(), i, k)

    elif case == 2:
        newtour = swapEdgesTwoOPT(tour.copy(), i, j)

    elif case == 3:
        newtour = swapEdgesTwoOPT(tour.copy(), j, k)

    elif case == 4:
        newtour = tour[:i + 1]
        newtour.extend(tour[j + 1:k + 1])
        newtour.extend(reversed(tour[i + 1:j + 1]))
        newtour.extend(tour[k + 1:])

    elif case == 5:
        newtour = tour[:i + 1]
        newtour.extend(reversed(tour[j + 1:k + 1]))
        newtour.extend(tour[i + 1:j + 1])
        newtour.extend(tour[k + 1:])

    elif case == 6:
        newtour = tour[:i + 1]
        newtour.extend(reversed(tour[i + 1:j + 1]))
        newtour.extend(reversed(tour[j + 1:k + 1]))
        newtour.extend(tour[k + 1:])

    elif case == 7:
        newtour = tour[:i + 1]
        newtour.extend(tour[j + 1:k + 1])
        newtour.extend(tour[i + 1:j + 1])
        newtour.extend(tour[k + 1:])

    return newtour


def heuristic_3opt_bi(positions):
    # tracking improvemnt in tour
    # print("heuristic_3opt_bi is called")
    improved = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    distances = np.rint(distances * 10000)
    distances = distances.astype(int)
    best_distance = route_distance(tour, distances)
    # tours: list with tours
    tours = []
    # swap_indices: list with indices to swap
    swap_indices = []
    while improved:
        improved = False
        for i in range(len(best_tour)):
            for j in range(i + 2, len(best_tour) - 1):
                for k in range(j + 2, len(best_tour) - 2 + (i > 0)):
                    # print(i, j, k)
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[j + 1]
                    e, f = tour[k], tour[k + 1]

                    # possible cases of removing three edges
                    # and adding three
                    deltacase = {
                        1: distances[a][e] + distances[b][f] \
                           - distances[a][b] - distances[e][f],

                        2: distances[a][c] + distances[b][d] \
                           - distances[a][b] - distances[c][d],

                        3: distances[c][e] + distances[d][f] \
                           - distances[c][d] - distances[e][f],

                        4: distances[a][d] + distances[e][c] + distances[b][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],

                        5: distances[a][e] + distances[d][b] + distances[c][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],

                        6: distances[a][c] + distances[b][e] + distances[d][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],

                        7: distances[a][d] + distances[e][b] + distances[c][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],
                    }

                    # get the case with most benefit
                    bestcase = min(deltacase, key=deltacase.get)
                    # print("bestcase ",bestcase)
                    new_tour = swapEdgesThreeOPT(tour.copy(), i, j, k, case=bestcase)
                    # print("new_tour ",new_tour)
                    new_distance = route_distance(new_tour, distances)
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_tour = new_tour
                        improved = True
                    # print("best_distance ", best_distance)

        tour = best_tour
    assert len(best_tour) == len(tour)

    best_tour = np.array(best_tour)
    tours = np.array(tours)

    return best_distance / 10000


def heuristic_3opt_fi(positions):
    # tracking improvemnt in tour
    print("positions ", positions)
    improved = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    distances = np.rint(distances * 10000)
    distances = distances.astype(int)
    best_distance = route_distance(tour, distances)
    # tours: list with tours
    tours = []
    # swap_indices: list with indices to swap
    swap_indices = []

    while improved:
        improved = False
        for i in range(len(best_tour)):
            for j in range(i + 2, len(best_tour) - 1):
                for k in range(j + 2, len(best_tour) - 2 + (i > 0)):
                    # print(i, j, k)
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[j + 1]
                    e, f = tour[k], tour[k + 1]

                    # possible cases of removing three edges
                    # and adding three
                    deltacase = {
                        1: distances[a][e] + distances[b][f] \
                           - distances[a][b] - distances[e][f],

                        2: distances[a][c] + distances[b][d] \
                           - distances[a][b] - distances[c][d],

                        3: distances[c][e] + distances[d][f] \
                           - distances[c][d] - distances[e][f],

                        4: distances[a][d] + distances[e][c] + distances[b][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],

                        5: distances[a][e] + distances[d][b] + distances[c][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],

                        6: distances[a][c] + distances[b][e] + distances[d][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],

                        7: distances[a][d] + distances[e][b] + distances[c][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],
                    }

                    # get the case with most benefit
                    bestcase = min(deltacase, key=deltacase.get)
                    # print("bestcase ",bestcase)
                    new_tour = swapEdgesThreeOPT(tour.copy(), i, j, k, case=bestcase)
                    # print("new_tour ",new_tour)
                    new_distance = route_distance(new_tour, distances)
                    # print("new_distance ", new_distance)
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_tour = new_tour
                        improved = True
                        break
                if improved:
                    break
        tour = best_tour
    assert len(best_tour) == len(tour)

    best_tour = np.array(best_tour)
    tours = np.array(tours)

    return best_distance / 10000


def heuristic_3opt_fi_restart(positions, steps):
    # tracking improvemnt in tour
    improved = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    distances = np.rint(distances * 10000)
    distances = distances.astype(int)
    best_distance = route_distance(tour, distances)
    restart_distance = best_distance

    for n in range(steps):
        improved = False
        for i in range(len(best_tour)):
            for j in range(i + 2, len(best_tour) - 1):
                for k in range(j + 2, len(best_tour) - 2 + (i > 0)):
                    # print(i, j, k)
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[j + 1]
                    e, f = tour[k], tour[k + 1]

                    # possible cases of removing three edges
                    # and adding three
                    deltacase = {
                        1: distances[a][e] + distances[b][f] \
                           - distances[a][b] - distances[e][f],

                        2: distances[a][c] + distances[b][d] \
                           - distances[a][b] - distances[c][d],

                        3: distances[c][e] + distances[d][f] \
                           - distances[c][d] - distances[e][f],

                        4: distances[a][d] + distances[e][c] + distances[b][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],

                        5: distances[a][e] + distances[d][b] + distances[c][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],

                        6: distances[a][c] + distances[b][e] + distances[d][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],

                        7: distances[a][d] + distances[e][b] + distances[c][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],
                    }

                    # get the case with most benefit
                    bestcase = min(deltacase, key=deltacase.get)
                    # print("bestcase ",bestcase)
                    new_tour = swapEdgesThreeOPT(tour.copy(), i, j, k, case=bestcase)
                    # print("new_tour ",new_tour)
                    new_distance = route_distance(new_tour, distances)
                    # print("new_distance ", new_distance)
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_tour = new_tour
                        improved = True
                        break
                if improved:
                    break
        tour = best_tour
        if improved is False:
            if best_distance < restart_distance:
                restart_distance = best_distance
                restart_tour = best_tour

            tour = create_tour(len(tour))
            best_distance = 1e10
        if n == steps - 1:
            if best_distance < restart_distance:
                restart_distance = best_distance
                restart_tour = best_tour
    assert len(best_tour) == len(tour)

    return best_distance / 10000


def heuristic_2opt_bi_restart(positions, steps):
    # tracking improvemnt in tour
    # print("heuristic_3opt_bi is called")
    improved = True
    tour = [x for x in range(len(positions))]
    best_tour = tour
    distances = calculate_distances(positions)
    distances = np.rint(distances * 10000)
    distances = distances.astype(int)
    best_distance = route_distance(tour, distances)
    restart_distance = best_distance
    # tours: list with tours
    tours = []
    # swap_indices: list with indices to swap
    swap_indices = []

    for n in range(steps):
        improved = False
        for i in range(len(best_tour)):
            for j in range(i + 2, len(best_tour) - 1):
                for k in range(j + 2, len(best_tour) - 2 + (i > 0)):
                    # print(i, j, k)
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[j + 1]
                    e, f = tour[k], tour[k + 1]

                    # possible cases of removing three edges
                    # and adding three
                    deltacase = {
                        1: distances[a][e] + distances[b][f] \
                           - distances[a][b] - distances[e][f],

                        2: distances[a][c] + distances[b][d] \
                           - distances[a][b] - distances[c][d],

                        3: distances[c][e] + distances[d][f] \
                           - distances[c][d] - distances[e][f],

                        4: distances[a][d] + distances[e][c] + distances[b][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],

                        5: distances[a][e] + distances[d][b] + distances[c][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],

                        6: distances[a][c] + distances[b][e] + distances[d][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],

                        7: distances[a][d] + distances[e][b] + distances[c][f] \
                           - distances[a][b] - distances[c][d] - distances[e][f],
                    }

                    # get the case with most benefit
                    bestcase = min(deltacase, key=deltacase.get)
                    # print("bestcase ",bestcase)
                    new_tour = swapEdgesThreeOPT(tour.copy(), i, j, k, case=bestcase)
                    # print("new_tour ",new_tour)
                    new_distance = route_distance(new_tour, distances)
                    # print("new_distance ", new_distance)
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_tour = new_tour
                        improved = True
        tour = best_tour
        if improved is False:
            if best_distance < restart_distance:
                restart_distance = best_distance
                restart_tour = best_tour

            tour = create_tour(len(tour))
            best_distance = 1e10
        if n == steps - 1:
            if best_distance < restart_distance:
                restart_distance = best_distance
                restart_tour = best_tour
    assert len(best_tour) == len(tour)

    best_tour = np.array(best_tour)
    tours = np.array(tours)

    return best_distance / 10000
