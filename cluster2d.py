'''
Note: Requires matplotlib pkg. To install in PyCharm:
File > Settings > Python Interpreter > + > (search for) matplotlib > install

Note: command line params: python cluster2d.py k max_iterations data_filename output_filename
k should be 3-5 usually
max_iter 10 is fine
data_filename Clustering_DataPoints.txt (large) or cluster.txt (small)
output_filename output.txt is fine


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

This script has been enhanced with a k-means++ initialization function. For more information, see
https://en.wikipedia.org/wiki/K-means%2B%2B

@TODO: The script should be enhanced to identify the optimal k-number using an Elbow algorithm.
See for example https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/

'''

import sys

import matplotlib.pyplot as plt


# return: each list entry is a tuple of a point in 2-d space. Ex: [(1.3, 2.2), ....]
def get_data_points(data_filename: str) -> []:
    import csv
    data_points = []
    with open(data_filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in reader:
            (x, y) = float(row[0]), float(row[1])
            data_points.append((x, y))
    # print("read from file: {}".format(str(data_points)))
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
    # print("x_min, x_max, y_min, y_max: {} {} {} {}".format(x_min, x_max, y_min, y_max))
    return x_min, x_max, y_min, y_max


# generates k random cluster values
# return: each list entry is a tuple of a point in 2-d space. Ex: [(1.3, 2.2), ....]
# DEPRECATED: Use generate_k_means_plus_plus_initial_cluster_centers() instead.
def generate_initial_cluster_centers(k: int, x_min: float, x_max: float, y_min: float, y_max: float) -> []:
    import random
    cluster_centers = []
    # random.seed()
    for i in range(k):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        cluster_centers.append((x, y))
        # print("Cluster center: ({}, {})".format(round(cluster_centers[i][0], 1), round(cluster_centers[i][1], 1)))
    return cluster_centers


def generate_k_means_plus_plus_initial_cluster_centers(k: int, data_points: []) -> []:
    '''
    Performs k-means++ initialization of the cluster centers.
    The first cluster center is assigned to a random data point. Then, each additional
    cluster center is assigned to one of the remaining data points. Each new cluster center
    is calculated by finding a remaining data point whose minimum distance from any existing cluster
    center is further than any other data point. For example, suppose you have 2 cluster centers
    and need one more. The first data point is distance 5 from the first cluster center, and
    is distance 10 from the second cluster center, so it gets assigned the minimum distance 5.
    All of the other remaining data point distances are also calculated, and the data point with
    the largest minimum distance becomes the next cluster center. When a data point is selected
    as a cluster center, it is removed from consideration when creating subsequent cluster centers.

    :param k: the number of cluster points to create
    :param data_points: list of 2D tuple data points
    :return: list of initial 2D tuple cluster centers
        Ex: [(1.3, 2.2), ....]
    '''
    import random

    cluster_centers = []
    # shallow copy
    remaining_data_points = data_points.copy()

    # choose initial random centroid from data points
    indx = random.randint(0, len(remaining_data_points) - 1)
    cluster_centers.append(remaining_data_points[indx])
    remaining_data_points.remove(remaining_data_points[indx])

    # for each remaining centroid
    for i in range(k - 1):
        # should never happen
        if not remaining_data_points:
            raise Exception("Unexpected ran out of data points")

        # minimum distances of each data point to all current centroids. number of entries
        # will be equal to number of data points
        minimum_distances = []

        # each remaining data point has distances to all current cluster centers.
        # Find the greatest of the least distances of each remaining data point

        # an initial distance. there is always at least one cluster center and at least one remaining data point
        for j in range(len(remaining_data_points)):
            minimum_distance = get_distance(remaining_data_points[j], cluster_centers[0])
            for k in range(len(cluster_centers)):
                minimum_distance \
                    = min(minimum_distance, get_distance(remaining_data_points[j], cluster_centers[k]))
            # the index of distances matches index of remaining data points
            minimum_distances.append(minimum_distance)
        maximum_distance = 0
        maximum_index = 0
        for j in range(len(minimum_distances)):
            if minimum_distances[j] > maximum_distance:
                maximum_distance = minimum_distances[j]
                maximum_index = j
        cluster_centers.append(remaining_data_points[maximum_index])
        remaining_data_points.remove(remaining_data_points[maximum_index])

    # print("Cluster center: ({}, {})".format(round(cluster_centers[i][0], 1), round(cluster_centers[i][1], 1)))
    return cluster_centers


def get_data_points_by_cluster(cluster_centers: [], data_points: []) -> {}:
    '''
    For each data point, determine which cluster it belongs to
    :param cluster_centers: list of 2D tuples for cluster points
    :param data_points: list of 2D tuples for data points
    :return: dict with key=cluster ordinal, value=list of 2D tuple data points belonging to that cluster point
            Ex: { 0: [ (1.1, 2.2), (1.2, 2.3), ...], ...}
    '''

    # create initialize the dict return value
    data_points_by_cluster = {}
    for i in range(len(cluster_centers)):
        data_points_by_cluster[i] = []
    for data_point in data_points:
        # there will always be at least one cluster point and data point
        closest_index = 0
        closest_distance = get_distance(data_point, cluster_centers[0])
        # find the cluster this data point is actually closest to
        for i in range(len(cluster_centers)):
            distance = get_distance(data_point, cluster_centers[i])
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i
        # print("adding data point to cluster {} ({}, {})".format(data_point,
        #                                                         round(cluster_centers[closest_index][0], 1),
        #                                                         round(cluster_centers[closest_index][1], 1)))
        data_points_by_cluster[closest_index].append(data_point)
    return data_points_by_cluster


def get_data_point_averages(cluster_centers, data_points_by_cluster: {}) -> []:
    '''
    Calculates the average position of all data points in a cluster, for all clusters
    :param cluster_centers: list of 2D tuples of cluster points
    :param data_points_by_cluster: dict with key=cluster point ordinal,
    value=list of 2D tuple data points belonging to the cluster
    :return: list of the 2D tuple positions of the averages of the points in each cluster.
    '''

    data_point_averages = []
    # indx is an ugly hack, just in case a cluster point somehow
    # ends up with no data points and we have to retain the original cluster point.
    # Probably impossible with k-means++, but probably could happen with plain k-means.
    indx = 0
    for key in sorted(data_points_by_cluster.keys()):
        count = len(data_points_by_cluster[key])
        if count:
            # If there are data points, add them up, calculate the mean, and update the output list
            x_sum = 0
            y_sum = 0
            for data_point in data_points_by_cluster[key]:
                x_sum += data_point[0]
                y_sum += data_point[1]
            x_avg = x_sum / count
            y_avg = y_sum / count
            data_point_averages.append((x_avg, y_avg))
        else:
            # keep the original cluster point
            data_point_averages.append(cluster_centers[indx])
        # keep track of the cluster center index
        indx += 1
    # print("data point averages:")
    # for avg in data_point_averages:
    #     print("({}, {})".format(round(avg[0], 1), round(avg[1], 1)))
    return data_point_averages


def any_clusters_changed(cluster_centers: [], data_point_averages: []) -> bool:
    '''
    Checks to see whether the cluster centers have converged yet. If not, the cluster_centers
    are updated to contain the new centers.
    :param cluster_centers: list of 2D tuples of cluster points. Size and order matches data_point_averages
    :param data_point_averages: list of average position of data points in the clusters.
    Size and order matches cluster_centers
    :return: True if any cluster center has changed, otherwise False
    '''
    clusters_changed = False
    for i in range(len(cluster_centers)):
        # print("before: ({}, {})".format(round(cluster_centers[i][0], 1), round(cluster_centers[i][1], 1)))
        if cluster_centers[i] != data_point_averages[i]:
            cluster_centers[i] = data_point_averages[i]
            clusters_changed = True
        # print("after: ({}, {})".format(round(cluster_centers[i][0], 1), round(cluster_centers[i][1], 1)))
    return clusters_changed


def report_results(data_points: [], cluster_centers: [], data_points_by_cluster: {}, title: str):
    '''
    Prints to a file a list of data points in the original input order, with the associated cluster ordinal.
    Not currently implemented. Instead, we write the output to stdout, and just for fun, display a plot
    of the results.
    :param data_points: list of 2D tuples of data points being analyzed
    :param cluster_centers: list of 2D tuples of cluster points for this data
    :param data_points_by_cluster: dict of data points assigned to clusters. key=cluster ordinal,
    value=list of data point 2D tuples for this cluster.
    '''

    # it is nice to print in input order, to make it easier to compare input to output.
    # To do this, we have to look up the cluster in data_points_by_cluster dict for each data point in data_points list
    # for i in range(len(data_points)):
    #     found = False
    #     for cluster, cluster_data_points in data_points_by_cluster.items():
    #         for cluster_data_point in cluster_data_points:
    #             if data_points[i][0] == cluster_data_point[0] and data_points[i][1] == cluster_data_point[1]:
    #                 print("{}\t{}\t{}".format(data_points[i][0], data_points[i][1], cluster + 1))
    #                 found = True
    #                 break
    #         if found:
    #             break
    #     if not found:
    #         # should never happen
    #         raise Exception("Cluster not found for {}".format(data_points[i]))
    #     # @TODO: write to file

    # Just for fun. Comment out if your connection does not support graphic display (e.g. ssh session) or
    # if you don't have the matplotlib library installed.from
    # This code only works correctly if k <= 10.
    # @TODO: enhance to work with any k value
    plt.style.use('seaborn-whitegrid')
    shapes = ["v", "^", "<", ">", "1", "2", "3", "4", "8", "s"]
    colors = ['#ff0000', '#00ff00', '#0000ff', "#888800", "#880088", "#008888", "#ee8844", "#88ee44", "#4488ee",
              "#44ee88"]
    # cluster colors are red circle, blue circle, and green circle.
    # cluster points are black <, black >, and black ^ shapes
    plt.title(title)
    for k, v in data_points_by_cluster.items():
        x_vals = []
        y_vals = []
        for x, y in v:
            x_vals.append(x)
            y_vals.append(y)
        plt.scatter(x_vals, y_vals, c=colors[k % 10], marker='o')
    for cluster_center in cluster_centers:
        plt.scatter(cluster_center[0], cluster_center[1], c='k', marker='o')
    # plt.ylabel('some numbers')
    plt.show()


def calculate_silhouette(data_points: [], cluster_centers: [], data_points_by_cluster: {}) -> float:
    '''
    Calculates the mean silhouette value for every data point for this k value. The higher the number,
    the better fitted are the data points to this number of cluster points.

    Using silhouette to find optimal k:
    sample = one of the data points
    a = average distance of sample to other points in this cluster
    b = average distance of sample to other points in the closest cluster that is not this cluster
    In a well grounded cluster, b - a is large
    Normalize by calculating (b-a)/max(a,b)
    Do this for every datapoint, and take the mean. That is the silhouette score for that k.
    The max Silhouette score is the optimal k.

    :param data_points: list of 2D tuples of data points being analyzed
    :param cluster_centers: list of 2D tuples of cluster points for this data
    :param data_points_by_cluster: dict of data points assigned to clusters. key=cluster ordinal,
    value=list of data point 2D tuples for this cluster.
    :return the mean silhouette value
    '''

    # this is a list of k silhouette values
    silhouettes = []
    # i is the current cluster center index
    for i in range(0, len(data_points_by_cluster)):
        # mean distances within this cluster.
        # The nth entry is the average distance from the nth data point
        # to all of the other points in this cluster
        # This is the 'a' value for the data points in this cluster
        a_values = []
        # mean distances for each point to the nearest external cluster.
        # The nth entry is the average distance from the nth data point
        # to all of the other points in the nearest external cluster
        # This is the 'b' value for the data points in this cluster
        b_values = []

        # calculate the nearest other cluster to this cluster
        nearest_cluster = 0
        nearest_cluster_idx = 0
        first_nearest_cluster = True
        for x in range(0, len(cluster_centers)):
            # don't measure distance to self
            if x == i:
                continue
            distance = get_distance(cluster_centers[x], cluster_centers[i])
            if first_nearest_cluster:
                first_nearest_cluster = False
                nearest_cluster = distance
                nearest_cluster_idx = x
            else:
                if distance < nearest_cluster:
                    nearest_cluster = distance
                    nearest_cluster_idx = x
        silhouettes_for_this_cluster = []
        for j in range(0, len(data_points)):
            #  find mean distance to other points in this cluster
            distances = []
            for k in range(0, len(data_points)):
                # don't check point against itself
                if j == k:
                    continue
                distance = get_distance(data_points[j], data_points[k])
                distances.append(distance)
            mean_distance_a = sum(distances) / len(distances)
            a_values.append(mean_distance_a)
            # get the b value
            distances = []
            for p in range(0, len(data_points_by_cluster[nearest_cluster_idx])):
                distance = get_distance(data_points[j], data_points_by_cluster[nearest_cluster_idx][p])
                distances.append(distance)
            mean_distance_b = sum(distances) / len(distances)
            b_values.append(mean_distance_b)
            # a_value - b_value / max(a_value, b_value) is the silhouette for this data point
            silhouettes_for_this_cluster.append((mean_distance_a - mean_distance_b))
            #  / max(mean_distance_b, mean_distance_a))
        silhouettes.append(sum(silhouettes_for_this_cluster) / len(silhouettes_for_this_cluster))
    silhouette = sum(silhouettes) / len(silhouettes)
    return silhouette

def get_distance(data1, data2):
    '''
    Utility function to get the cartesian distance between two 2D data points
    :param data1: first data point
    :param data2: second datapoint
    :return: cartesian distance
    '''
    x_square = pow(data1[0] - data2[0], 2)
    y_square = pow(data1[1] - data2[1], 2)
    distance = pow(x_square + y_square, .5)
    return distance


def perform_cluster_analysis(k: int, max_iterations: int, data_filename: str) -> ():
    '''
    Performs all of the processing for k-means++.

    :param k: int - number of cluster points for this data. A cluster point is a 2D point in the dataspace
        that is in the center of a visually identifiable cluster.
    :param max_iterations: int - Number of convergence attempts. If the cluster points ever stop changing,
        that means they have converged to separate clusters. This is the max number of tries to reach
        convergence.
    :param data_filename: str - The name of a file that contains the 2D data points
    :param output_filename: str - The name of the output file. Will be overwritten on each execution.
    :return: tuple of cluster list, data_points list, and data_points_by_cluster dict.
    '''
    # print("Performing cluster analysis")

    # get input data
    # data_points is a list of 2D data point tuples
    data_points = get_data_points(data_filename)

    # This is used for vanilla k-means. It does not reliably find the optimal visually identifiable clusters.
    # Uncomment this, and comment out the other cluster_centers assignment function, if you want to try it out.
    # (x_min, x_max, y_min, y_max) = get_min_and_max_values(data_points)
    # cluster_centers = generate_initial_cluster_centers(k, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    # Use the k-mean++ algorithm to set the initial set of k cluster centers, using the input data points
    # cluster_centers is a list of cluster 2D tuples
    cluster_centers = generate_k_means_plus_plus_initial_cluster_centers(k, data_points)

    # See if the cluster centers will converge (stop changing on each iteration)
    for i in range(max_iterations):
        # find the data points close to the current list of cluster center
        # data_points_by_cluster is a dict with key= cluster ordinal int, value = list of data point 2D tuples
        # cluster ordinal int means the index of a particular cluster center in the cluster_center list
        data_points_by_cluster = get_data_points_by_cluster(cluster_centers=cluster_centers, data_points=data_points)

        # calculate the average position of the data points in each cluster
        # data_point_averages is a list of floats in cluster-center order
        data_point_averages = get_data_point_averages(cluster_centers=cluster_centers,
                                                      data_points_by_cluster=data_points_by_cluster)

        # check for convergence
        if not any_clusters_changed(cluster_centers, data_point_averages):
            # Perform the output and return
            # print("Converged after {} iterations".format(i))
            # write_output_file(data_points=data_points, cluster_centers=cluster_centers,
            #                   data_points_by_cluster=data_points_by_cluster)
            return cluster_centers, data_points, data_points_by_cluster
    # failed to converge
    print("Exceeded max iterations - no clusters found")
    return None


def main():
    '''
    Calculates the k-means cluster points for an input dataset. For convenience, a matplotlib plot
    of the clustered points is included. For now, it only works correctly with k=3.

    Even with k-means++ the winning cluster set only occurs about 70% of the time, so the analysis is
    performed an arbitrary number of times, and the most frequent result is displayed. Kind of a
    kludge, but it yields a good result, at least for this dataset.


    This script includes a main() function with these command line parameters:

    k: int - number of cluster points for this data. A cluster point is a 2D point in the dataspace
        that is in the center of a visually identifiable cluster.
    max_iterations: int - Number of convergence attempts. If the cluster points ever stop changing,
        that means they have converged to separate clusters. This is the max number of tries to reach
        convergence.
    data_filename: str - The name of a file that contains the 2D data points
    output_filename: str - The name of the output file. Will be overwritten on each execution.
    '''

    # Gets the input with minimal validation, just make sure the required params are all present
    cmd_line_params = sys.argv
    if len(cmd_line_params) != 5:
        print("Usage: python cluster2d.py k max_iterations, data_filename, output_filename")
        print("You used {}".format(str(sys.argv)))
        return
    k = int(sys.argv[1])
    max_iterations = int(sys.argv[2])
    data_filename = sys.argv[3]
    output_filename = sys.argv[4]

    silhouettes = []
    for k in range(2, 8):
        cluster_centers, data_points, data_points_by_cluster = perform_cluster_analysis(k=k,
                                                                                        max_iterations=max_iterations,
                                                                                        data_filename=data_filename)
        report_results(data_points, cluster_centers, data_points_by_cluster, "K: " + str(k))

        silhouette = calculate_silhouette(data_points, cluster_centers, data_points_by_cluster)
        silhouettes.append(silhouette)
        print("K: " + str(k) + " Silhouette value: " + str(silhouette))

    best_silhouette = 0
    best_silhouette_idx = 0
    for i in range(2, 8):
        if silhouettes[i-2] > best_silhouette:
            best_silhouette = silhouettes[i-2]
            best_silhouette_idx = i

    print("Best Silhouette: " + str(best_silhouette_idx))
    cluster_centers, data_points, data_points_by_cluster = \
        perform_cluster_analysis(k=best_silhouette_idx,
                                 max_iterations=max_iterations,
                                 data_filename=data_filename)
    report_results(data_points, cluster_centers, data_points_by_cluster, 'Winner K: ' + str(best_silhouette_idx))


if __name__ == "__main__":
    main()
