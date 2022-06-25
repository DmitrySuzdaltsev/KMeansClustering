# -*- coding: utf-8 -*-
"""
@author: Dmitry Suzdaltsev

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time


def euclidean_distance(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return np.sqrt(np.sum((x1-x2)**2))


def plot(X, labels, centroids, n_clusters):
    for i in range(n_clusters):
        color = (mpl.colors.hsv_to_rgb(((i+1)/n_clusters, 1, 1)))
        x, y, *_ = X[labels==i].T
        plt.scatter(x, y, marker='.', color=color)
        x, y, *_ = centroids[i]
        plt.scatter(x, y, marker='*', color=color, edgecolor='black', linewidth=1, s=200, alpha=0.4)
    plt.show()
    time.sleep(0.5)


def kmeans(dots, n_clusters, max_step=100):
    n_dots, n_features = dots.shape
    
    # init_centroids
    idx = np.random.choice(n_dots, n_clusters, replace=False)
    cluster_centroids = dots[idx]    
    
    for step in range(max_step):    
        # update distances
        distances = [[euclidean_distance(centroid, dot) for centroid in cluster_centroids] for dot in dots]
        # update centers
        cluster_labels = np.argmin(distances, axis=1)
        centers = [np.mean(dots[cluster_labels==label], axis=0) for label in range(n_clusters)]
        
        plot(dots, cluster_labels, centers, n_clusters)        
        if euclidean_distance(centers, cluster_centroids) < 1e-17:
            break
        else:
            cluster_centroids = centers
    return {'centroids': cluster_centroids, 'labels': cluster_labels}


if __name__ == '__main__':
    dots = np.random.rand(1500, 2)
    n_clusters = 5
    clusters = kmeans(dots, n_clusters)
    plot(dots, clusters['labels'], clusters['centroids'], n_clusters)
