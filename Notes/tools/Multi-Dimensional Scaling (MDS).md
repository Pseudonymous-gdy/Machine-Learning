
The Proximity Matrix of a matrix like distance matrix, similarity matrix, and dissimilarity matrix, remains unchanged by shifting, rotating, or reflection. And the basic idea of MDS is to keep the most of the proximity matrix.

# Classical MDS

## Mathematical Inference
Given a Proximity Marix $D\in\mathbb{R}^{n\times n}$, where $d_{ij}$ is the distance between samples $i$ and $j$. Then we hope to find an $X\in\mathbb{R}^{n\times k}$ such that $\|x_i-x_j\|\approx d_{ij}$. With the proximation, we have:$$\begin{aligned}
d_{ij}^2&=\|x_i-x_j\|^2=x_i^Tx_i+x_j^Tx_j-2x_i^Tx_j:=b_{ii}+b_{jj}-2b_{ij}\\
\text{where }B&=(b_{ij})_{i,j\in\{1,2,\cdots,n\}}=XX^T
\end{aligned}$$Since $D$ remains for any rotation, shift or reflection of $X$, we could consider a set of shifts and rotations of $X$ to get a determined result. The constraint for $X$ is$$\sum x_i=0$$
Then by calculation, we have$$\begin{aligned}D^{(2)}:&=(d_{ij}^2)=(b_{ii}+b_{jj}-2b_{ij})\\
\sum^n_{i=1}d_{ij}^2&=\text{Tr}(B)+nb_{jj}\\
\sum^n_{i=1}\sum^n_{j=1}d_{ij}^2&=2n\text{Tr}(B)\\
\Rightarrow b_{ij}&=-\frac{1}{2}\left(d_{ij}^2-\text{Avg}(d_{i \cdot}^2)-\text{Avg}(d_{\cdot j}^2)+\text{Avg}(d_{\cdot\cdot}^2)\right)\\
\Rightarrow B&=-\frac{1}{2}JD^{(2)}J, \text{where } J=I-\frac{1}{n}\mathbf{1}\mathbf{1}^T
\end{aligned}$$
## Steps
Thus, the basic steps for classical MDS are:
1. Calculate B through the formula above
	- Compute the squared proximity matrix $D^{(2)}=(d_{i j}^{2})$
	- Compute the inner product matrix B using double centering: $B=-\frac{1}{2} J D^{(2)} J$ , where $J=I-\frac{1}{n} 1 1^{\top}$ (double centering)
2. Conduct Spectral Decomposition of $B=P\Lambda P^{T}$ to compute $X=P\Lambda^{1/2}$.
	- Compute the k largest eigenvalues $\lambda_{1}, ..., \lambda_{k}$ and corresponding eigenvectors $e_{1}, ..., e_{k}$ of B
	- Obtain the projected coordinates $X=E_{k} \Lambda_{k}^{1 / 2}$ , where $E_{k}=[e_{1}, ..., e_{k}] \in \mathbb{R}^{n ×k}$ and $\Lambda_{k}^{1 / 2}=diag(\sqrt{\lambda_{1}}, ..., \sqrt{\lambda_{k}}) \in \mathbb{R}^{k ×k}$.

# Strain Function

We apply `Strain Function` to evaluate the quality of the MDS solution, which is given by$$\text{Strain}(x_1,x_2,\cdots,x_n)=\sqrt{\sum_{i<j}\left(b_{ij}-x_i^Tx_j\right)^2/\sum_{i<j}b_{ij}^2}$$with lower strain as better fit. This helps us fit the parameter $k$ since Classical MDS has already given a closed-form optimal solution.

# Metric MDS

In classical MDS, the proximity matrix is primarily given by Euclidean Distance measurement, and thus possesses limitations in generalization. Probably, we hope our algorithm to fit:
- More general proximity matrix, e.g., based on dissimilarity
- Support different weights between points

Therefore, here, we start up with the metric first, which is `Stress Function` partially based on `Strain Function`$$\text{Stress}\left(x_{1}, ..., x_{n}\right)=\sqrt{\sum_{i<j} w_{i j}\left(d_{i j}-\left\| x_{i}-x_{j}\right\| \right)^{2} / \sum_{i<j}\left\| x_{i}-x_{j}\right\| ^{2}}$$or sometimes$$\text{Stress}\left(x_{1}, ..., x_{n}\right)=\sqrt{\sum_{i<j}\left(d_{i j}-\left\| x_{i}-x_{j}\right\| \right)^{2} / \sum_{i<j} d_{i j}^{2}}$$
Please note that Metric MDS does not have a closed form solution, therefore, the algorithm is realized through optimizing the `Stress` function iteratively. Also, it is called Stress majorization.

# Non-Metric MDS

In some cases, the measurement of proximity matrix are qualitative, where we only focus on their rank, or namely the ordering results of dissimilarity in low-dimensional space should be consistent with the high-dimensional space. Then, for a further step of generalization, the metric `Stress function` is given by$$\text{Stress}\left(x_{1}, ..., x_{n}, \textcolor{red}{f}\right)=\sqrt{\sum_{i<j}\left(f\left(d_{i j}\right)-\left\| x_{i}-x_{j}\right\| ^{2}\right) / \sum_{i<j}\left\| x_{i}-x_{j}\right\| ^{2}}$$where $f$ is a monotonically increasing function, i.e., $d_{ij}<d_{kl}\Leftrightarrow f(d_{ij})<f(d_{kl})$.

Furthermore, we need a generalized algorithm, which is:
- Use monotonic regression to found optimal $f$ to best minimizing the stress function
- Optimize $x_1,x_2,\cdots,x_n$ to best minimizing the stress function
- Repeat the two steps alternatively

# Pros & Cons

1. Pros
	- Preserves pairwise distances as closely as possible
	- Flexibly works with any dissimilarity matrix
2. Cons
	- Assumes linear relationships, faling to capture nonlinear manifold structures (variant: [[Isomap]])
	- Computationally intensive for large datasets
	- Sensitive to noise and outliers in the dissimilarity matrix
	- Results depend on the choice of distance metric and initialization