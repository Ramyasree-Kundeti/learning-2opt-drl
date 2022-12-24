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


def swapEdgesTwoOPT(tour, i, j):
    """
		Method to swap two edges and replace with
		their cross.
		"""
    newtour = tour[:i + 1]
    newtour.extend(reversed(tour[i + 1:j + 1]))
    newtour.extend(tour[j + 1:])

    return newtour


def swapEdgesThreeOPT(tour, i, j, k):
    """
	Method to swap edges for Segment Shift
	"""
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


                    new_tour = swapEdgesThreeOPT(tour.copy(), i, j, k)
                    # print("new_tour ",new_tour)
                    new_distance = route_distance(new_tour, distances)
                    print("new_distance ", new_distance)
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_tour = new_tour
                        improved = True
        tour = best_tour
    assert len(best_tour) == len(tour)

    best_tour = np.array(best_tour)
    tours = np.array(tours)

    return best_distance / 10000


def heuristic_3opt_fi(positions):
    # tracking improvemnt in tour
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

                    new_tour = swapEdgesThreeOPT(tour.copy(), i, j, k)
                    # print("new_tour ",new_tour)
                    new_distance = route_distance(new_tour, distances)
                    print("new_distance ", new_distance)
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

                    new_tour = swapEdgesThreeOPT(tour.copy(), i, j, k)
                    # print("new_tour ",new_tour)
                    new_distance = route_distance(new_tour, distances)
                    print("new_distance ", new_distance)
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

                    new_tour = swapEdgesThreeOPT(tour.copy(), i, j, k)
                    # print("new_tour ",new_tour)
                    new_distance = route_distance(new_tour, distances)
                    print("new_distance ", new_distance)
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
        if n == steps-1:
            if best_distance < restart_distance:
                restart_distance = best_distance
                restart_tour = best_tour
    assert len(best_tour) == len(tour)

    best_tour = np.array(best_tour)
    tours = np.array(tours)

    return best_distance / 10000