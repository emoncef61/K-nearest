import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Define data points and a new point to classify
points = {"blue": [[2, 4], [1, 3], [3, 2], [2, 1]],
          "red": [[5, 6], [4, 6], [6, 4], [4, 4]]}
new_point = [3, 3]

def euclidean_distance(p, q):
    """
    Calculate the Euclidean distance between two points.

    :param p: First point as a list [x, y].
    :param q: Second point as a list [x, y].
    :return: Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2)

class KNearestNeighbors:
    def __init__(self, k=3):
        """
        Initialize a KNearestNeighbors instance.

        :param k: The number of nearest neighbors to consider (default is 3).
        :type k: int
        """
        self.k = k
        self.points = {}

    def fit(self, points):
        """
        Fit the KNearestNeighbors model to the provided data points.

        :param points: Data points with categories.
        :type points: dict
        """
        self.points = points

    def predict(self, new_point):
        """
        Predict the category of a new data point.

        :param new_point: The data point to classify.
        :type new_point: list [x, y]
        :return: The predicted category for the new point.
        """
        distances = []

        for category, category_points in self.points.items():
            for point in category_points:
                dis = euclidean_distance(new_point, point)
                distances.append((category, dis))

        categories = [category[0] for category in sorted(distances, key=lambda x: x[1])[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
        return result

# Create a KNearestNeighbors classifier and fit it to the provided data
clf = KNearestNeighbors()
clf.fit(points)

# Visualize the data points and the classification result
ax = plt.subplot()
ax.grid(True, color="#323232")
ax.set_facecolor("black")
ax.figure.set_facecolor("#121212")
ax.tick_params(axis="x", color="white")
ax.tick_params(axis="y", color="white")

# Scatter plot for blue points
for point in points['blue']:
    ax.scatter(point[0], point[1], color="#104DCA", s=60)

# Scatter plot for red points
for point in points['red']:
    ax.scatter(point[0], point[1], color="#FF0000", s=60)

# Predict the class of the new point and plot it with the appropriate color
new_class = clf.predict(new_point)
color = "#FF0000" if new_class == 'red' else "#104DCA"
ax.scatter(new_point[0], new_point[1], color=color, marker='*', s=200, edgecolor='black', linewidth=1)
plt.show()
