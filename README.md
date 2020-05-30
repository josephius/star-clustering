# Star Clustering

## Introduction

The Star Clustering algorithm is a clustering technique that is loosely inspired and analogous to the process of star system formation.  Its purpose is as an alternative clustering algorithm that does not require knowing the number of clusters in advance or any hyperparameter tuning.

## Installation

The following dependencies should be installed:

* numpy

## Setup

It is recommended that you have Scikit-Learn as the implementation provided here is meant to work with Scikit-Learn as a drop in replacement for other algorithms.

The actual algorithm is located in star_clustering.py and can be called by the import statement:

`from star_clustering import StarCluster`

Then create an object to instantiate an instance of the algorithm:

`star = StarCluster()`

Then call the fit or predict functions as you would any other clustering algorithm in Scikit-Learn.

## Test Scripts

Three test scripts are provided that are meant to show the effectiveness of the algorithm on very different types of data.

* plot_cluster_comparison.py
* word_vectors_test.py
* plot_cluster_iris.py

Note that the word vectors test requires a copy of the FastText pretrained word vectors or some equivalent, which is not provided here.

## Example Plot Results

### Performance On Various Synthetic Test Data

![Plot Cluster Comparison - Star Clustering](Figure_StarClustering.png)

### Comparison To Other Algorithms

![Plot Cluster Comparison - Star Clustering](Figure_Plot_Cluster_Comparison.png)

### Performance On Iris Data

![Plot Cluster Iris - Star Clustering](Figure_Iris_Star_Clustering.png)

### Iris Data Ground Truth

![Plot Cluster Iris - Ground Truth](Figure_Iris_Ground_Truth.png)

## Licence

Apache 2.0