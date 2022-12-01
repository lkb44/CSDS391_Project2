import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

def plot_data(d, is_title=False, title=''):
    for r in d:
        # Assigning colors to species
        if r[4] == 'setosa':
            c = 'red'
        elif r[4] == 'versicolor':
            c = 'green'
        elif r[4] == 'virginica':
            c = 'blue'
        else:
            c = 'black'
        # Plot Species
        plt.plot(float(r[2]), float(r[3]), linestyle='none', marker='o', color=c)
    # Add plot labels
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    if is_title:
        plt.title(title)
    pass

def k_means_cluster(k, d):
    # Initialize initial means as k different random points
    averages = k * [[0.0, 0.0]]
    for index in range(k):
        # Check that two initial means can't be the same starting point
        while True:
            random_point = random.randint(0, len(d))
            if not averages.__contains__([float(d[random_point][2]), float(d[random_point][3])]):
                break
        averages[index] = [float(d[random_point][2]), float(d[random_point][3])]
    # Initialize objective function
    objective_function = []
    # Initialize iterations
    iterations = 0
    while True:
        iterations += 1
        if len(d) == 150:
            # Plot the data against averages
            plot_data(d)
            for average in averages:
                plt.plot(average[0], average[1], color='black', marker='o', linestyle='none')
            # Obtain data point for objective function
            objective_function.append(get_objective_function(d, averages))
            # Show the graph
            plt.title('k-means clustering for ' + str(k) + ' Clusters\nIteration: ' + str(iterations))
            plt.show()
        # Check that averages change with each iteration
        last_averages = averages.copy()
        # Categorize each data point with its closest mean
        averages = [[0.0, 0.0] for i in range(k)]
        num_points = k * [0.0]
        for r in d:
            # Classify points to closest averages
            index = last_averages.index(get_closest_mean([r[2], r[3]], last_averages))
            # Count points assiciates to averages
            num_points[index] += 1.0
            # Assign x and y positions to the average
            averages[index][0] += float(r[2])
            averages[index][1] += float(r[3])
        # Make the sums an average by dividing by the number of points that are closest to a specific average
        for index in range(k):
            averages[index][0] /= num_points[index]
            averages[index][1] /= num_points[index]

        # If the cluster points did not shift locations, return the clusters
        if last_averages == averages:
            if len(d) == 150:
                # Plot Objective Function
                plt.plot(range(len(objective_function)), objective_function, 'ko')
                plt.plot(range(len(objective_function)), objective_function, 'k')
                plt.xlabel('Iteration')
                plt.ylabel('Sum of Error Squared')
                plt.title('Objective Function for ' + str(k) + ' Clusters')
                plt.show()

                # Return the centroids of the clusters
                return [averages, objective_function]
            else:
                return [averages, 0]


def get_objective_function(d, means):
    sse = 0.0

    for r in d:
        # Error at point squared = ||xn - uk||^2
        sse += math.pow(get_distance([float(r[2]), float(r[3])], get_closest_mean([float(r[2]), float(r[3])], means)), 2)

    return sse


def get_decision_bounds(point1, point2, t):
    # Intercept point (halfway between the two points in the parameters)
    x_constant = (point1[0] + point2[0]) / 2.0
    y_constant = (point1[1] + point2[1]) / 2.0

    # Slope of the line (if y = mx + b for the line )
    slope = abs(point2[0] - point1[0]) / abs(point2[1] - point1[1])
    return y_constant - (t - x_constant) / slope


def get_likelihood(point, cluster, clusters):
    # distance_test is 1 / the distance from the point to the cluster
    distance_test = 1.0 / get_distance(point, cluster)

    # distance_all is 1 / the sum of the distance from the point to each cluster
    distance_all = 0.0
    for c in clusters:
        distance_all += 1.0 / get_distance(point, c)

    # return the percentage likelihood of it being a specific cluster at a specific point
    return distance_test / distance_all


def plot_decision_boundaries(num_clusters, iris_data, t):
    uk = []
    d = []

    # Get clusters
    output = k_means_cluster(num_clusters, iris_data)
    uk.append(output[0])
    d.append(output[1])
    plot_data(iris_data)

    for index in range(len(uk[0])):
        p1 = [uk[0][index][0], uk[0][index][1]]
        plt.plot(p1[0], p1[1], 'ko', linestyle='none')

        for index2 in range((index + 1), len(uk[0])):
            p2 = uk[0][index2][0], uk[0][index2][1]

            x = (p1[0] + p2[0]) / 2.0
            y = (p1[1] + p2[1]) / 2.0

            # line
            m = abs(p2[0] - p1[0]) / abs(p2[1] - p1[1])
            line = y - (t - x) / m

            # Determine if the line should be plotted
            l1 = get_likelihood([x, y], p1, uk[0])
            l3 = 1.0 - (2.0 * l1)
            if l3 < l1:
                plt.plot(t, line, 'c:')

    # Make plot fancier and show the plot
    names = ['setosa', 'versicolor', 'virginica', 'Cluster', 'Decision Boundaries']
    colors = ['r', 'g', 'b', 'k', 'c']
    hands = []

    for i in range(5):
        hands.append(mpatches.Patch(color=colors[i], label=names[i]))

    plt.legend(handles=hands, loc='upper left')

    if len(iris_data) == 150:
        plt.xlim(0.0, 7.1)
        plt.ylim(0.0, 2.6)
    else:
        plt.xlim(2.9, 7.1)
        plt.ylim(0.9, 2.6)

    plt.title('Decision Boundaries for ' + str(num_clusters) + ' Clusters')
    plt.show()

    pass


def get_closest_mean(r, means):
    # returns the mean that is closest to the point
    distance = 99.9
    closest_mean = [0.0, 0.0]

    for mean in means:
        # Get distance between the current point and the mean
        d = get_distance([float(r[0]), float(r[1])], mean)

        # If current average is closer to the point than the prior average, make the prior average the current average
        if d < distance:
            distance = d
            closest_mean = [float(mean[0]), float(mean[1])]

    return closest_mean

def get_distance(pa, pb):
    # returns the distance between 2 points
    # return abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
    return math.sqrt(math.pow(pa[0] - pb[0], 2) + math.pow(pa[1] - pb[1], 2))

with open('irisdata.csv') as file:
    # Used to take out the header from the file
    heading = next(file)

    # iris data
    iris = csv.reader(file)
    data = []

    # Get the iris data stored as variable data
    for row in iris:
        # Add the iris data to the data 2D array
        data.append(row)
    t = np.linspace(0.0, 7.0, 200)

    # Exercises: 1a, 1b, 1c, and 1d for k = 2
    plot_decision_boundaries(2, data, t)
    # Exercises: 1a, 1b, 1c, and 1d for k = 3
    plot_decision_boundaries(3, data, t)
