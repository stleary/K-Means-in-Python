'''
Introduction to Computing in the Life Sciences – BOT 576/576 MCB 576
Detailed Project Descriptions

Clustering Algorithm Implementation—given a set of data points in two dimensions, use a
provided algorithm to cluster the data into N related groups.

Goal of the algorithm:
Given a set of 2-dimensional data points (x, y), group the data points together into k groups
(where k is defined by the user) so that each group represents a “cluster”—that is, a set
containing points that are spatially near to each other.

Algorithm:
Initialize k cluster centers – choose each center to be a data point (x, y) where x is a uniform
random draw (use the random module’s uniform function) from the x-range of the data points
and y is a uniform random draw from the y-range of the data points. For example, if x_min is
the minimum x value observed among all data points, and x_max is the maximum x value
observed among all data points, then the x coordinate for each center would be drawn from the
interval [x_min, x_max].
[Brief note on random: Documentation for the random module can be found here:
https://docs.python.org/3/library/random.html
Sample use of the uniform function:
import random
draw_float = random.uniform(4.2, 6.7)
]

While not converged:
--Assign each datapoint (x,y) to the closest cluster center (x_c, y_c) (using Euclidean distance).
--Recompute each cluster center as the average of the points assigned to that cluster.
Convergence is reached when the assignment of points to clusters does not change after an
iteration.
(The user will specify a maximum number of iterations beyond which the program stops even if
convergence has not been reached; if this occurs, the program will write output and print to the
screen that the maximum number of iterations has been reached.)
Concepts:
Cluster centers are just points—they are separate from the data points, they represent an
‘average’ point in the cluster. If there are k clusters, then there are always k cluster centers.
Compute the average of a set of points by averaging their x values and averaging their y values.
The Euclidean distance between two points (x, y) and (p,q) is: sqrt( (x – p)^2 + (y – q)^2 )
(This is not Python code, it’s just a formula).
Implementation:

The program will take four arguments from the command line:
1. k: The number k of clusters
2. max_iterations: The maximum number of iterations allowed
3. data_filename: The name of the data points file
4. output_filename: The name of the output file

The data points file will be a tab-delimited file defining a list of data observations associated
with two coordinates. Example as follows:
1.4 6.8
5.6 9.8
3.2 9.7

The program will create a single output file which assigns data points to clusters, example as
follows (suppose k in this case is 3), please include the file header as shown:
X Y Cluster
1.4 6.8 1
5.6 9.8 2
3.2 9.7 2
2.4 5.8 1
6.6 3.8 3
8.2 5.2 2
9.4 6.6 3
2.6 2.1 3
1.2 3.7 1
3.2 4.3 1

A program plot_clusters.R will be provided to you along with a sample calling script
call_plot_clusters.sh, in order to help visualize the output of your program. You are
welcome to use these plots in debugging and in your presentation.
Naming:

Please place the final version of your program in your ICLS/project folder and name it
cluster2d.py
===========================================================================================

'''


# cmd line params: k, max_iterations, data_filename, output_filename
# k = 3, max_iterations = ?, data_filename=input.txt output_filename=output.txt


# return: each list entry is a tuple of a point in 2-d space. Ex: [(1.3, 2.2), ....]
def get_data_points(data_filename: str) -> []:
    import csv
    data_points = []
    with open(data_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            (x, y) = float(row[0]), float(row[1])
            data_points.append((x, y))
    print("read from file: {}".format(str(data_points)))
    return data_points


# calculates min and max values
def get_min_and_max_values(data_points: []) -> (float, float, float, float):
    x_min = data_points[0][0]
    y_min = data_points[0][1]
    x_max = data_points[0][0]
    y_max = data_points[0][1]
    for data_point in data_points:
        if data_point[0] < x_min:
            x_min = data_point[0]
        if data_point[0] > x_max:
            x_max = data_point[0]
        if data_point[1] < y_min:
            y_min = data_point[1]
        if data_point[1] > y_max:
            y_max = data_point[1]
    print("x_min, x_max, y_min, y_max: {} {} {} {}".format(x_min, x_max, y_min, y_max))
    return x_min, x_max, y_min, y_max


# generates k random cluster values
# return: each list entry is a tuple of a point in 2-d space. Ex: [(1.3, 2.2), ....]
def generate_initial_cluster_centers(k: int, x_min: float, x_max: float, y_min: float, y_max: float) -> []:
    import random
    cluster_centers = []
    # random.seed()
    for i in range(k):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        cluster_centers.append((x, y))
        print("Cluster center: ({}, {})".format(round(cluster_centers[i][0], 1), round(cluster_centers[i][1], 1)))
    return cluster_centers


# generates k-means++ random cluster values
# return: each list entry is a tuple of a point in 2-d space. Ex: [(1.3, 2.2), ....]
def generate_k_means_plus_plus_initial_cluster_centers(k: int, data_points: []) -> []:
    import random

    cluster_centers = []
    # shallow copy
    remaining_data_points = data_points.copy()

    # choose initial random centroid from data points
    indx = random.randint(0, len(data_points)-1)
    cluster_centers.append(data_points[indx])
    remaining_data_points.remove(data_points[indx])

    # for each remaining centroid
    for i in range(k-1):
        # should never happen
        if not remaining_data_points:
            raise Exception("Unexpected ran out of data points")

        # minimum distances of each data point to all current centroids. number of entries
        # will be equal to number of data points
        distances = []

        # each remaining data point has distances to all current cluster centers.
        # Find the least distance for each remaining data point

        # an initial distance. there is always at least one cluster center and at least one remaining data point
        distance = get_distance_to_cluster_center(data_point=remaining_data_points[0], cluster_center=cluster_centers[0])
        for j in range(len(remaining_data_points)):
            for k in range(len(cluster_centers)):
                distance = min(distance, get_distance_to_cluster_center(data_point=remaining_data_points[j], cluster_center=cluster_centers[k]))
            # the index of distances matches index of remaining data points
            distances.append(distance)
        # find the furthest distance
        indx = 0
        min_distance = distances[0]
        for j in range(len(distances)):
            if distances[j] < min_distance:
                indx = j
                min_distance = distances[j]
        cluster_centers.append(remaining_data_points[indx])
        remaining_data_points.remove(remaining_data_points[indx])

    # print("Cluster center: ({}, {})".format(round(cluster_centers[i][0], 1), round(cluster_centers[i][1], 1)))
    return cluster_centers


def get_distance_to_cluster_center(data_point: (), cluster_center: ()) -> float:
    x_square = pow(cluster_center[0] - data_point[0], 2)
    y_square = pow(cluster_center[1] - data_point[1], 2)
    distance = pow(x_square + y_square, .5)
    # print("datapoint cluster distance {} {} {}".format(data_point, cluster_center, distance))
    return distance


# return: each dict key is a cluster_center index.
# each dict valueis a list of tuples of 2-d points.
# Ex: { 0: [ (1.1, 2.2), (1.2, 2.3), ...], ...}
def get_data_points_by_cluster(cluster_centers: [], data_points: []) -> {}:
    data_points_by_cluster = {}
    # intialize the dict
    for i in range(len(cluster_centers)):
        data_points_by_cluster[i] = []
    for data_point in data_points:
        closest_index = 0
        closest_distance = get_distance_to_cluster_center(data_point, cluster_centers[0])
        for i in range(len(cluster_centers)):
            distance = get_distance_to_cluster_center(data_point, cluster_centers[i])
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i
        print("adding data point to cluster {} ({}, {})".format(data_point,
                                                                round(cluster_centers[closest_index][0], 1),
                                                                round(cluster_centers[closest_index][1], 1)))
        data_points_by_cluster[closest_index].append(data_point)
    return data_points_by_cluster


# calculates the average position of all data points in a cluster, for all clusters
# return: each list entry is an average of the points in a cluster.
# the list size is the same and ordering as cluster_centers (k)
# These will become the new cluster points
# ex: [(1.5, 2.5), ....]
def get_data_point_averages(cluster_centers, data_points_by_cluster: {}) -> []:
    data_point_averages = []
    # indx is an ugly hack
    indx = 0
    for key in sorted(data_points_by_cluster.keys()):
        count = len(data_points_by_cluster[key])
        x_avg = 0
        y_avg = 0
        if count:
            x_sum = 0
            y_sum = 0
            for data_point in data_points_by_cluster[key]:
                x_sum += data_point[0]
                y_sum += data_point[1]
            x_avg = x_sum / count
            y_avg = y_sum / count
            data_point_averages.append((x_avg, y_avg))
        else:
            # using the ugly hack
            data_point_averages.append(cluster_centers[indx])
        indx += 1
    print("data point averages:")
    for avg in data_point_averages:
        print("({}, {})".format(round(avg[0], 1), round(avg[1], 1)))
    return data_point_averages


# returns True if any cluster center does not equal the corresponding data_point_average
def any_clusters_changed(cluster_centers: [], data_point_averages: []) -> bool:
    clusters_changed = False
    for i in range(len(cluster_centers)):
        print("before: ({}, {})".format(round(cluster_centers[i][0], 1), round(cluster_centers[i][1], 1)))
        if cluster_centers[i] != data_point_averages[i]:
            cluster_centers[i] = data_point_averages[i]
            clusters_changed = True
        print("after: ({}, {})".format(round(cluster_centers[i][0], 1), round(cluster_centers[i][1], 1)))
    return clusters_changed


# prints to a file a list of data points in the original input order, with the associated cluster ordinal.
# Use tabs to delimit
# for each data point, search for it in the cluster centers list
def write_output_file(data_points: [], cluster_centers: [], data_points_by_cluster: {}):
    for i in range(len(data_points)):
        found = False
        for cluster, cluster_data_points in data_points_by_cluster.items():
            for cluster_data_point in cluster_data_points:
                if data_points[i][0] == cluster_data_point[0] and data_points[i][1] == cluster_data_point[1]:
                    print("{}\t{}\t{}".format(data_points[i][0], data_points[i][1], cluster + 1))
                    found = True
                    break
            if found:
                break
        if not found:
            print("Cluster not found for {}".format(data_points[i]))
        # write to file

    # just for fun
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    colors = ['ro', 'bo', 'go']
    for k, v in data_points_by_cluster.items():
        color = colors[k]
        x_vals = []
        y_vals = []
        for x, y in v:
            x_vals.append(x)
            y_vals.append(y)
        plt.plot(x_vals, y_vals, color)
    plt.plot(cluster_centers[0][0], cluster_centers[0][1], 'k<')
    plt.plot(cluster_centers[1][0], cluster_centers[1][1], 'k>')
    plt.plot(cluster_centers[2][0], cluster_centers[2][1], 'k^')
    plt.ylabel('some numbers')
    plt.show()


# return value is a tuple of cluster list, data_points list, and dict.
# list is cluster values, dict key=cluster ordinal value=datapoints
def perform_cluster_analysis(k: int, max_iterations: int, data_filename: str, output_filename: str) -> ():
    print("Performing cluster analysis")
    data_points = get_data_points(data_filename)
    (x_min, x_max, y_min, y_max) = get_min_and_max_values(data_points)
    # cluster_centers = generate_initial_cluster_centers(k, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    cluster_centers = generate_k_means_plus_plus_initial_cluster_centers(k, data_points)
    for i in range(max_iterations):
        data_points_by_cluster = get_data_points_by_cluster(cluster_centers=cluster_centers, data_points=data_points)
        data_point_averages = get_data_point_averages(cluster_centers=cluster_centers,
                                                      data_points_by_cluster=data_points_by_cluster)
        if not any_clusters_changed(cluster_centers, data_point_averages):
            print("Converged after {} iterations".format(i))
            write_output_file(data_points=data_points, cluster_centers=cluster_centers,
                              data_points_by_cluster=data_points_by_cluster)
            return cluster_centers, data_points, data_points_by_cluster
    print("Exceeded max iterations - no clusters found")
    return None


def main():
    import sys
    cmd_line_params = sys.argv
    if len(cmd_line_params) != 5:
        print("Usage python cluster2d.py k max_iterations, data_filename, output_filename")
        print("You used {}".format(str(sys.argv)))
        return
    k = int(sys.argv[1])
    max_iterations = int(sys.argv[2])
    data_filename = sys.argv[3]
    output_filename = sys.argv[4]

    # results is a dict {total_distance: (cluster_centers, data_points, data_points_by_cluster)}
    results = {}
    for i in range(10):
        cluster_centers, data_points, data_points_by_cluster = perform_cluster_analysis(k=k,
                                                                                        max_iterations=max_iterations,
                                                                                        data_filename=data_filename,
                                                                                        output_filename=output_filename)
    #     indx = 0
    #     total_distance = 0
    #     for cluster_center in cluster_centers:
    #         inner_indx = 0
    #         for data_point in data_points_by_cluster[indx]:
    #             distance = get_distance_to_cluster_center(data_point, cluster_center)
    #             total_distance += distance
    #             inner_indx += 1
    #         # normalize by number of items
    #         avgtotal_distance /= inner_indx
    #         results[total_distance] = (cluster_centers, data_points, data_points_by_cluster)
    #         indx += 1
    #
    # keys = sorted(results.keys())
    # key = keys[0]
    # cc, dp, dpbc = results[key]
    # write_output_file(data_points=dp, cluster_centers=cc,
    #                   data_points_by_cluster=dpbc)


if __name__ == "__main__":
    main()
