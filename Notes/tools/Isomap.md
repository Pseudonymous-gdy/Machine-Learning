
For example, the distribution on a manifold below might not be successfully handled by Euclidean distance in the context of [[Multi-Dimensional Scaling (MDS)]]. 
![[Pasted image 20250806191249.png]]
Therefore, we might consider further exploration on a manifold instead of mere Euclidean distance, which leads to Isomap, which uses geodesic distances (shortest paths on the manifold) instead of Euclidean distances to capture nonlinear structures.

# Steps

1. Construct Neighborhood Graph: Build a graph where each data point is connected to its $k$ nearest neighbors or points within radius $\epsilon$.
2. Compute Geodesic Distances: Estimate geodesic distances between all pairs of points using shortest paths (e.g., Dijkstra’s algorithm).
3. Apply MDS: Use classical MDS to embed the data into a lower-dimensional space, preserving geodesic distances.
4. Output: Low-dimensional coordinates that reflect the manifold’s geometry.


# Pros & Cons

1. Pros
	- Captures nonlinear manifold structures effectively
	- Preserves global geometric properties via geodesic distances
2. Cons
	- Sensitive to the choice of $k$ or $\epsilon$ for neighborhood graph
	- Computationally expensive for large datasets (shortest path calculations)
	- Assumes data lies on a single, well-sampled manifold
	- May fail if the manifold has holes or is noisy.