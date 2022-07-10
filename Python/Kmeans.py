# -*- coding: utf-8 -*-
"""
Created on Sun May 22 19:13:48 2022

@author: user
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np



Z = np.load('C:/Project-Zeev/angl/s3/save_dir/save_dir/990_1.npy', mmap_mode='r')
Y = np.load('C:/Project-Zeev/angl/s3/save_dir/save_dir/990_0.npy', mmap_mode='r')
X = np.concatenate((Z, Y), axis=0)
#dist=np.load('dist.npy')
num_clusters = 2
#cluster the data into k clusters, specify the k  
kmeans = KMeans(n_clusters = num_clusters)
kmeans.fit(X)
labels = kmeans.labels_ + 1
#show the clustering results  

plt.bar(range(len(labels)),labels) 
plt.title("The partitions cluster labels")
plt.xlabel("The number of partition")
plt.ylabel("The cluster label")
plt.show()

# calculate the silhouette values  
silhouette_avg = silhouette_score(X, labels)
sample_silhouette_values = silhouette_samples(X, labels)
# show the silhouette values 
plt.plot(sample_silhouette_values) 
plt.title("The silhouette plot")
plt.xlabel("The number of partition")
plt.ylabel("The silhouette coefficient values")

xmin=0
xmax=len(labels)
# The vertical line for average silhouette score of all the values
plt.hlines(silhouette_avg, xmin, xmax, colors='red', linestyles="--") 
plt.show()

_ = plt.hist(X, bins='auto')
plt.show()

