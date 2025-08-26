
# SNE
The goal of this algorithm is to preserve the neighborhood information of high-dimensional data in a low-dimensional embedding. This is realized through modelling pairwise similarities as probabilities in both high- and low-dimensional spaces.

## Mathematical Inference

We define high-dimensional similarity (a point jump from j to i in the original space) as$$p_{j | i}=\frac{exp \left(-\left\| x_{i}-x_{j}\right\| ^{2} / 2 \sigma_{i}^{2}\right)}{\sum_{k \neq i} exp \left(-\left\| x_{i}-x_{k}\right\| ^{2} / 2 \sigma_{i}^{2}\right)}$$where:
- $p_{i|i}=0$
- $\sigma_i$ is a bandwidth parameter, representing the "size" of the neighborhood
- Search $\sigma_i$ such that $H(P_i)=-\sum_jp(j|i)\log p_{j|i}=\log\text{Perplexity}$

Then for low-dimensional similarity (a point jump from j to i in the embedded space) is given by$$q_{j|i}=\frac{\exp\left(-\|y_i-y_j\|^2\right)}{\sum_{k\neq i}\exp\left(-\|y_i-y_j\|^2\right)}$$where $\sigma_i$ is defaultedly set. This helps to
- reduce the computational complexity
- avoids overfitting to the high-dimensional structure

Then, we wish to minimize the KL divergence between the two distributions, which is therefore researched as$$\begin{aligned}
C&=\sum_i\text{KL}(P_i\|Q_i)=\sum_i\sum_jp_{j|i}\log\frac{p_{j|i}}{q_{j|i}}\\
\frac{\partial C}{\partial y_i}&=2\sum_j(y_i-y_j)(p_{i|j}-q_{i|j}+p_{j|i}-q_{j|i})
\end{aligned}$$where we could see that the optimization enables
- connection via "springs" between every two samples
- a forced direction of $y_i-y_j$
- a forced magnitude of $p_{i|j}-q_{i|j}+p_{j|i}-q_{j|i}$, which is the mismatch between high-dimensional similarity and low-dimensional similarity

## Pros & Cons

1. Pros
	- Preserves local structure effectively
	- flexible framework for extensions
2. Cons
	- Crowding problem: points clump in low-dimensional space
	- Computationally expensive for large datasets
	- Sensitive to perplexity parameter

# t-SNE

## Intro: Crowding Problem

There are two facts about crowding problem in the previous setting:
1. high-dimensional data enjoys higher degree of freedom and geometric capacity, then certain adoptable geometric structure in high-dimensional structure might not be well incorporated into lower dimensional data. (e.g., the equal distance between points of a triangle might not be well kept in one-dimensional map)
2. distribution of pairwise distances in high-D space and low-dimensional space is totally different, with $V(r)\propto r^D$. This indicates fewer space for low dimension to place neighborhood points, making the mapping to be "crowded"

Thus, samples with medium distance with sample $i$ will be placed too far away (e.g., $p_{j|i}>q_{j|i}$) contributing to a large number of small attractive forces compared to $p_{j|i}$. Then, sample $i$ will be pulled to the center of the map. This could be realized by increasing $q_{ij}$ for distant points, which leads to t-distribution. t-distribution enables
- A long-tail distribution to convert distances into probabilities
- This allows a moderate distance in the high-dimensional space to be faithfully modeled by a much larger distance on the map
- It eliminates the unwanted attractive forces between map points that represent moderately dissimilar datapoints.

## Changed mathematical formulation

High-dimensional similarity (symmetrized): $$p_{i j}=\frac{p_{j | i}+p_{i | j}}{2 n}$$
Low-dimensional similarity (t-distribution, 1 degree of freedom): $$q_{i j}=\frac{\left(1+\left\| y_{i}-y_{j}\right\| ^{2}\right)^{-1}}{\sum_{k \neq l}\left(1+\left\| y_{k}-y_{l}\right\| ^{2}\right)^{-1}}$$
Objective: Minimize KL divergence: $$C=KL(P \| Q)=\sum_{i \neq j} p_{i j} \log \frac{p_{i j}}{q_{i j}}$$
To minimize the objective function, we applied early exaggeration, which is a technique to enhance cluster formation. This helps get the results in clearer, more interpretable visualizations with well-defined clusters.

`Early Exaggeration`

Purpose: helps from tight distinct clusters early in the optimization

Process:
1. Start with exaggerated similarities to pull similar points closer.
2. After ∼100 iterations, use true similarities to fine-tune positions.

## Hyperparameter Tuning

`Perplexity`: controls the balance between local and global structure
- this is corresponded to the effective number of neighbors used to compute conditional probabilities.
- Low perplexity (e.g., 5–10): Emphasizes local structure, producing tight, fine-grained clusters but potentially missing global relationships.
- High perplexity (e.g., 30–50): Captures more global structure, spreading out points and potentially merging nearby clusters.

`Learning rates`: Determines the step size during gradient descent optimization of the KL divergence.

## Pros & Cons

1. Pros
	- Excellent for visualizing clusters
	- Mitigates crowding problem via t-distribution
	- Robust to different datasets
2. Cons
	- Non-convex optimization (sensitive initialization)
	- Does not preserve global structure well
	- Computationally intensive for large datasets
	- No easy way to embed new points