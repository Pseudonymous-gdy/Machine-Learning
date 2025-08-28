
# Agglomerative Hierarchical Clustering

## Procedure

- Start with each data point as a single cluster.
- Merge the closest pairs of clusters iteratively.
- Continue until one cluster remains.

## Linkage Methods: how to measure the distance between two clusters?

For example, Ward's method: Cost of merge cluster A and B is represented as $$\Delta(A, B)=\sum_{x \in A \cup B}\left\| x-\mu_{A \cup B}\right\| ^{2}-\sum_{x \in A}\left\| x-\mu_{A}\right\| ^{2}-\sum_{x \in B}\left\| x-\mu_{B}\right\| ^{2}=\frac{2 n_{A} n_{B}}{n_{A}+n_{B}}\left\| \mu_{A}-\mu_{B}\right\| ^{2}$$Other methods are listed below.

| Linkage Method   | Distance Measure                                   | Characteristics                                                                                                                                                                                                                      | Suitable Scenarios                                                                                                        |
| ---------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| Single Linkage   | Minimum pairwise distance between two clusters     | - Tends to form **elongated or chain-like clusters** (due to "chaining effect").  <br>- Highly sensitive to noise and outliers (noise can connect unrelated clusters).  <br>- Captures weak connections between sparse data regions. | Data with naturally elongated structures (e.g., geographic clusters along a river).                                       |
| Complete Linkage | Maximum pairwise distance between two clusters     | - Produces **compact, spherical clusters** (avoids overly large clusters).  <br>- Less sensitive to noise than single linkage but may split natural clusters.  <br>- Tends to emphasize cluster boundaries.                          | Data with distinct, dense, and roughly spherical clusters (e.g., well-separated groups in physics experiments).           |
| Average Linkage  | Average of all pairwise distances between clusters | - Balances between single and complete linkage (reduces chaining and over-compaction).  <br>- Produces more **balanced cluster sizes**.  <br>- Moderately sensitive to noise.                                                        | General-purpose clustering where neither extreme (elongated nor overly compact) is desired (e.g., customer segmentation). |
| Ward’s Method    | Increase in total within-cluster variance (SS)     | - Minimizes the sum of squared differences within clusters (similar to k-means).  <br>- Produces **tight, homogeneous clusters** with minimal internal variance.  <br>- Computationally efficient for large datasets.                | Scenarios requiring clusters with minimal internal spread (e.g., gene expression data analysis, quality control groups).  |

# Divisive Hierarchical Clustering

## Procedure

- Start with all data points in one cluster.
- Split the cluster into smaller clusters iteratively.
	- e.g. by finding the two most dissimilar points in the cluster and using them to separate the data into two parts.
- Continue until each data point is in its own cluster.
- Less commonly used than agglomerative clustering.
	- Much more computationally intensive

## Dendrogram

Dendrogram is a tree-like diagram that shows the arrangement of clusters, with `Height` representing the distance between clusters (calculated by linkage methods). It is useful for visualizing hierarchical clustering results, which could be cut at a certain height to determine the number of clusters.