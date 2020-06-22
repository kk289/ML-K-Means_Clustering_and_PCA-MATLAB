# Machine Learning (MATLAB) - *K*-Means Clustering and Principle Component Analysis

Machine Learning course from Stanford University on [Coursera](https://www.coursera.org/learn/machine-learning/home/week/8).

### Introduction
We will implement the K-means clustering algorithm and apply it to compress an image. And will use principal component analysis to find a low-dimensional representation of face images.

### Environment
- macOS Catalina (version 10.15.3)
- MATLAB 2018 b

### Dataset
- ex7data1.mat
- ex7data2.mat
- ex7faces.mat

### Files included in this repo
- ex7.m - Octave/MATLAB script for the K-means clustering 
- ex7_pca.m - Octave/MATLAB script for PCA 
- ex7data1.mat - Example Dataset for PCA
- ex7data2.mat - Example Dataset for K-means
- ex7faces.mat - Faces Dataset
- bird_small.png - Example Image
- displayData.m - Displays 2D data stored in a matrix
- drawLine.m - Draws a line over an exsiting figure 
- plotDataPoints.m - Initialization for K-means centroids 
- plotProgresskMeans.m - Plots each step of K-means as it proceeds
- runkMeans.m - Runs the K-means algorithm
- submit.m - Submission script that sends our solutions to the servers 

[⋆] pca.m - Perform principal component analysis

[⋆] projectData.m - Projects a data set into a lower dimensional space 

[⋆] recoverData.m - Recovers the original data from the projection 

[⋆] findClosestCentroids.m - Findclosestcentroids(usedinK-means) 

[⋆] computeCentroids.m - Compute centroid means (used in K-means) 

[⋆] kMeansInitCentroids.m - Initialization for K-means centroids

## K-mean Clustering
We will implement the K-means algorithm and use it for image compression. We will first start on an example 2D dataset that help to gain an intuition of how k-menas algorithm works. After that, we wil use the K-means algorithm for image compression by reducing the number of colors that occur in an image to only those that are most common in that image.

we will use following script for this part:
```
ex7.m
```

### Part 1.1: Implementing K-means
The K-means algorithm is a method to automatically cluster similar data examples together. Concretely, we are given a training set {x^(1),...,x^(m)} (where x^(i) ∈ R^n), and want to group the data into a few cohesive “clusters”.

#### Part 1.1.1: Finding closest centroids
```
% Load an example dataset
load('ex7data2.mat');
```

##### findClosestCentroids.m
This function takes the data matrix X and the locations of all centroids inside centroids and should output a one-dimensional array idx that holds the index (a value in {1,...,K}, where K is total number of centroids) of the closest centroid to every training example. We can implement this using a loop over every training example and every centroid.

```
function idx = findClosestCentroids(X, centroids)

% Set K
K = size(centroids, 1);

% return the following variables correctly.
idx = zeros(size(X,1), 1);

for i = 1:size(X,1)
   temp = (X(i,:) - centroids);
   [a idx(i)] = min(sum((temp) .* (temp),2));
end
end
```

Result: 
Closest centroids for the first 3 examples: 1 3 2

#### Part 1.1.2: Computing centroid means

##### computeCentroids.m
```
function centroids = computeCentroids(X, idx, K)

% Useful variables
[m n] = size(X);

% to return the following variables correctly.
centroids = zeros(K, n);

% for loop to compute centroid means
for k = 1:K
  count = 0;
  sum = zeros(n, 1);
  for i = 1:m
    if (idx(i) == k)
      sum = sum + X(i, :)';
      count = count+1;
    end
  end
  centroids(k, :) = (sum/count)';
end

end
```

Result: 

Computing centroids means

Centroids computed after initial finding of closest centroids:  
[2.428301 3.157924]   
[5.813503 2.633656]   
[7.119387 3.616684]   


## Course Links 

1) Machine Learning by Stanford University on [Coursera](https://www.coursera.org/learn/machine-learning/home/week/8).

2) [K-Means Clustering and PCA](https://www.coursera.org/learn/machine-learning/home/week/8)
(Please notice that you need to log in to see the programming assignment.) #ML-K-Means_Clustering_and_PCA-MATLAB