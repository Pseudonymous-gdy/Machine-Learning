# K-Means

![[Pasted image 20250731224038.png]]

For `stopping conditions`, we have:
- Maximum iterations reached
- Assignments no longer change
- Convergence to local optimum of cost $\mathcal{L}$

While the last condition mentioned cost function, we might discuss it and the roles related.

## Cost Function

The cost function is given by$$\mathcal{L}(\Delta)=\sum^K_{k=1}\sum_{i\in C_k}\|x_i-\mu_k\|^2$$Which indicates a least-squares problem, and the cost function is called $\textcolor{red}{\text{Quadratic Distortion}}$ (calculating distance using Euclidean Distance, which highlights the level of dispersion).

Moreover, K-Means realize reduction on this cost function every step, since:
- Reassigning the labels can only decrease $\mathcal{L}$
- Reassigning the centers $\mu_k$ can only decrease $\mathcal{L}$

## Optimal number of clusters (Hyperparameter Tuning)

Here we adopt Elbow method, which is a heuristic method to determine the optimal number of clusters $k$.

`Steps`:
- Run k-means clustering for a range of k values
- Calculate the Within-Cluster Sum of Squares (WCSS) (which is $\mathcal{L}$) for each k
- Plot WCSS against k.
- Look for an "elbow" point where adding more clusters yields diminishing returns.

`Goal`: Balance between minimizing WCSS and avoiding overfitting with too many clusters.

## Pros and Cons

1. Pros
	- Simple and Fast
	- Work Well with `Spherical Clusters`.
2. Cons
	- Sensitive to Initial Centers
	- Assume Clusters are of Similar Size and Density

This leads us to further thinking about how to improve K-means, from which we hope to start a `smart initialization` of the algorithm.

## Improving K-Means

Basically for random K-Means initialization, with the increase of number $k$, we have decreasing probability to hit all $k$ clusters with $k$ samples. Therefore we hope to intentionally generate the initialized points so that **points are far away from each other to have space to develop**. We will introduce three ways, which are `FFT`, `k-Means++`, and also `K-log K initialization`.

### Fastest First Traversal (FFT)

This is a greedy algorithm to simply spread out centers. Basically, it conducts an algorithm as:
- While $|C|<k$:
	- Find $T=\{t|\min_{x\in C}d(t,x)\}$
	- Find $x^*=\max(T)$
	- $C\gets C\cup\{x^*\}$
While it is simple, such greedy algorithm might not guarantee a stable, near-optimal initialization of centers. Then we introduce `k-Means++`.

### k-Means++

This is a randomized, theoretically backed approach to spread out centers. The key idea is to spread centers by favoring points farther from existing ones, balancing randomness and determinism. The algorithm is given as:
1. Choose the first center randomly from the data points.
2. For each remaining point, compute its squared distance to the nearest existing center.
3. Select the next center with probability proportional to that squared distance.
4. Repeat until K centers are chosen.

Then, we have the bound of: $E[\phi_{K- Means ++}] ≤8(\ln K+2) \cdot \phi_{OPT}$, where $\phi$ represents the error.

`Advantages`: Reduces sensitivity to poor initialization, theoretically guarantees $O(log K)$-competitive solution.

### K-log K

This is a initializat method starting with enough centers ($K\log K$) to hit all clusters, then prune down to $K$.

Then for pruning, the per-step goal is to minimize the loss function$$\mathcal{L}(\Delta)=\sum_{k=1}^{K} \sum_{i \in C_{k}}\left\| x_{i}-\mu_{k}\right\| ^{2}$$even though two centers are combined.

# K-Medoids

Similar to k-means but use actual data points (medoids) as cluster centers. Here, `Medoid` is calculated through minimizing the same loss function with K-Means, while `Medoid` is an existing data point compared to centroid. Also, the procedure assembles, which is given by:
- Initialize k medoids randomly
- Assign each point to the nearest medoid
- Update medoids to minimize the total cost
- Repeat until convergence

Such method, compared to K-Means, have its own pros and cons.
1. Pros:
	- Robust to noise and outliers
	- Works with any distance metric
2. Cons:
	- Computationally expensive for large datasets, which is `NP-Hard`.

# Kernel Tricks

Due to the brief calculation in [[Kernel Trick#Prototype-based Method]], for every centroid $\mu_k$, we have $$\varphi(\mu_k)=\frac{1}{|C_k|}\sum_{x_i\in C_k}\varphi(x_i)$$And$$\|\varphi(x_i)-\varphi(x_j)\|^2=K(x_i,x_i)-2K(x_i,x_j)+K(x_j,x_j):=K_{ii}-2K_{ij}+K_{jj}$$
Therefore, we get$$\begin{aligned}
\|\varphi(x_j)-\varphi(\mu_k)\|^2&=\|\varphi(x_j)-\frac{1}{|C_k|}\sum_{x_i\in C_k}\varphi(x_i)\|^2\\
&=K_{jj}-\frac{2}{|C_k|}\sum_{x_i\in C_k}K_{ji}+\frac{1}{|C_k|^2}\sum_{x_m,x_n\in C_k}K_{mn}\quad(*)
\end{aligned}$$
Which leads to the procedure of Kernel K-Means:
1. Initialize: Randomly assign points to k clusters.
2. Assign: For each point $x_{i}$ , compute distance to each cluster’s centroid using $(*)$, and assign $x_i$ to the closest centroid.
3. Update: Recompute kernel sums for each cluster based on new assignments.
4. Repeat: Until assignments stabilize.