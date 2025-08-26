
**Document Class**  #Clustering #Evaluation #introduction #DSAA2011

**Table of Contents:**
- [[#Definition|Definition]]
- [[#Metrics|Metrics]]
	- [[#Metrics#External Metrics|External Metrics]]
		- [[#External Metrics#Purity|Purity]]
		- [[#External Metrics#(Adjusted) Rank Index|(Adjusted) Rank Index]]
		- [[#External Metrics#Fowlkes-Mallows index|Fowlkes-Mallows index]]
		- [[#External Metrics#(Normalized) Mutual Information|(Normalized) Mutual Information]]
		- [[#External Metrics#V -measure|V -measure]]
	- [[#Metrics#Internal Metrics|Internal Metrics]]
		- [[#Internal Metrics#Silhouette Score|Silhouette Score]]
		- [[#Internal Metrics#Calinski-Harabaz Index (Variance Ratio Criterion)|Calinski-Harabaz Index (Variance Ratio Criterion)]]
		- [[#Internal Metrics#Davies-Bouldin Index|Davies-Bouldin Index]]
		- [[#Internal Metrics#Dunn index|Dunn index]]
- [[#Methods|Methods]]
	- [[#Methods#(Hard) Prototype-based|(Hard) Prototype-based]]
	- [[#Methods#(Hard) Density-based|(Hard) Density-based]]
	- [[#Methods#(Hard) Hierarchical|(Hard) Hierarchical]]
	- [[#Methods#Soft Clustering|Soft Clustering]]

**A Guide on 1$^\mathbf{st}$ Unsupervised Learning.**

---
## Definition

Clustering, in short, is to find groups in data, which is:
- Given a dataset $\mathcal{D}$ consist of $n$ data points
- We want to separate them into $K$ clusters
	- $\Delta=\{C_1,C_2,...,C_K\}:\text{ partition of }\mathcal{D}$
	- $\mathcal{L}(\mathcal{\Delta}):\text{ cost (loss function) of }\Delta$

## Metrics

Then the problem goes to: is a clustering good or not? To answer this question, we have to introduce different metrics for us to evaluate the effectiveness of a clustering method in the context of the dataset. Then the metrics goes to two kinds of thoughts, which are:
- How much does clustering contribute to the supervised learning.
- How well are data points themselves clustered.
Which leads to two different kinds of metrics.

1. External Metrics
	This kind of metrics use external information, e.g., the category of images, to measure how well the clustering results match the ground-truth labels.
	
	Please notice that it is different from supervised learning since the labels are not utilized for training but for evaluation.
2. Internal Metrics
	Do not use any external information, and assess the goodness of clusters based only on the intrinsic structure of the data. This kind of clustering methods primarily focus on:
	- Compactness of clusters (how close points are within a cluster)
	- Separation between clusters (how distinct clusters are from each other)

And we need many different metrics because:
- Diversity in scenarios
- Strengths and weaknesses for each metric
- Complexity of data
- Diversity in evaluation goals
### External Metrics

#### Purity

This measure is aimed at calculating the fraction of correctly classified points. In short, it could be written as:$$\frac{K}{n}\leq\text{Purity}=\frac{1}{n}\sum^K_{i=1}\max_j|C_i\cap L_j|\leq 1$$
Where $n$ is the total number of data points, $K$ is the number of clusters, $C_i$ is the points in cluster $i$, and $L_j$ is the points of label (category) $j$.

This encourages clustering to have higher purity, while it also encourages over-clustering (e.g., set every single point as a cluster)

`Example`:
```python
import numpy as np
def purity_score(y_true, y_pred):
	cm = contingency_matrix(y_true,y_pred)  
	return np.sum(np.max(cm,axis=0)) / len(y_true)
y-gt=np.repeat([0,1,2],10)
y1=np.repeat([0,1,2],10)
y1=np.concatenate(y1[3:],y1[:3]1)
y2=np.repeat([0,1,2,3,4,5],5)
y2=np.concatenate(y2[3:],y2[:311])
y3=np.array(range(30))
print(purityscore(y-gt,y1))#0.7 
print(purity_score(y-gt,y2))#0.8
print(purity_score(y-gt,y3))#1.0
```

#### (Adjusted) Rank Index

As an external metric, we wish the clustering to contribute to future supervised learning as much as possible. This makes measures on the equality of data points from the same cluster important. Basically, Rank Index set cases similar to confusion matrix, based on the hypothesis that same ground truth labels could be identified by machine if with same clustering labels.

|                       | Ground truth labels |  $l_i=l_j$  |  $l_i\neq l_j$   |
| :-------------------: | :-----------------: | :---------: | :--------------: |
| **Clustering labels** |        Types        | same labels | different labels |
|       $c_i=c_j$       |    same cluster     |     TP      |        FP        |
|     $c_i\neq c_j$     | different clusters  |     FN      |        TN        |
Then similar to the accuracy, the RI index is given by:$$\text{RI}=\frac{TP+TN}{TP+TN+FP+FN}=\frac{TP+TN}{n\choose 2}\in[0,1]$$
Also, a `higher RI` is preferred. Compared to [[#Purity]], this metric `punishes over-clustering` since when every data point is a cluster, lots of False Negative cases exist, leading to a lower RI value.

However, such metric doesn't consider the possibility of `chance agreements` between the two clusters. Namely, some clustering methods include certain randomness, which should not be taken into account (otherwise, random clustering might be of high RI values). And that leads to `Adjusted Rank Index (ARI)`. Basically, ARI, with `1 as the perfect match`, is given by$$\text{ARI}=\frac{\text{RI}-\text{Expected RI}}{\max(\text{RI})-\text{Expected RI}}\in[-1,1]$$
Where $\text{Expected RI}$ is specified by a `random model` measuring the `expected similarity` of all `pair-wise comparisons`. A <font color=green>contingency table</font> is used when calculating.

#### Fowlkes-Mallows index

This method, with the basis of [[#(Adjusted) Rank Index]], measures the geometric mean of precision and recall for pairs of points, which is given by$$\text{FMI}=\sqrt{\text{Precision}\cdot\text{Recall}}=\sqrt{\frac{TP}{TP+FP}\cdot\frac{TP}{TP+FN}}\in[0,1]$$
Similar to scores for classification tasks, a higher value of FMI is preferred, with 1 as the perfect match.

#### (Normalized) Mutual Information

Before showing the formula, we introduce entropy, conditional entropy, and mutual information.

`Entropy` quantifies average level of uncertainty: $H(X)=\mathbb{E}[-ln P(X)]=\sum_{x \in X}-P(x) ln P(x)$

`Conditional entropy` quantifies the amount of information needed to describe Y given that the value of X: $$\begin{aligned} H(Y | X)=\sum_{x \in X} P(X=x) H(Y | X=x) & =\sum_{x \in X} P(x) \sum_{y \in Y}-P(y | x) ln P(y | x) \\ & =-\sum_{x \in X, y \in Y} P(x,y) \ln\frac{P(x,y)}{P(x)} \end{aligned}$$

`Mutual information` quantifies the amount of information obtained about Y after observing X (and vice versa): $$\begin{aligned} I(X, Y)=I(Y, X) & =H(X)-H(X | Y) \\ & =H(Y)-H(Y | X) \\ & =H(X)+H(Y)-H(X, Y) \\ & =H(X, Y)-H(X | Y)-H(Y | X) \end{aligned}$$
Then let $X$ be the clustering results, $Y$ be the ground truth labels, and then $MI$ reflects the <font color=red>alignment between them</font>. For `Normalized Mutual Information`, with `1 as the perfect match`:$$\text{NMI}=\frac{2\cdot I(C,L)}{H(C)+H(L)}\in[0,1]$$
where $C$ are clusters, and $L$ are labels.

This algorithm excels at mining the relationships between clustering results and truth labels, and show the matching level between the two. However, it is sensitive to the distribution (worst when dataset unbalances), and costly in computational resources.

`Example`:
```python
import numpy as np
from sklearn.metrics.cluster import mutual_info_score, normalized_mutual_info_score

y_gt = np.repeat([0,1,2],10)
y1 = np.repeat([0,1,2],10)
y1 = np.concatenate([y1[3:],y1[:3]])
y2 = np.repeat([0,1,2,3,4,5],5)
y2 = np.concatenate([y2[3:],y2[:3]])
y3 = np.array(range(50))

print(mutual_info_score(y_gt,y_gt)
normalized_mutual_info_score(y_gt,y_gt)
print(mutual_info_score(y_gt,y1)
normalized_mutual_info_score(y_gt,y1))
print(mutual_info_score(y_gt,y2)
normalized_mutual_info_score(y_gt,y2))
print(mutual_info_score(y_gt,y3)
normalized_mutual_info_score(y_gt,y3))
```
#### V -measure

This measure is a harmonic mean of two complementary components:
- Homogeneity: Measures how pure each cluster is (all points in a cluster belong to a single class).
- Completeness: Measures how well all points of a class are assigned to the same cluster.

With the basis of entropy, the formula is given by:$$\begin{aligned}Homogeneity =1-\frac{H(C | L)}{H(C)}, \quad Completeness =1-\frac{H(L | C)}{H(L)}\\V -measure =2 \cdot \frac{ Homogeneity \cdot Completeness }{ Homogeneity + Completeness }\in[0,1]\end{aligned}$$
Where 1 is seen as the perfect match. This algorithm exceeds at balancing between homogeneity and completeness and being less sensitive to unbalanced distributions. However, harmonic mean might lead to ignorance of bad performance in certain dimension.

### Internal Metrics

#### Silhouette Score

This method measures how similar a point is to its own cluster compared to other clusters. For a single point and the dataset, the formula is given by: $$\begin{aligned}s(i)=\frac{b(i)-a(i)}{max \{a(i), b(i)\}}&=\left\{\begin{array}{ll}1-a(i) / b(i) & a(i)<b(i) \\ a(i) / b(i)-1 & a(i) \geq b(i)\end{array} \in[-1,1]\right.\\ S&=\frac{1}{n}\sum_is(i)\end{aligned}$$
Where:
- $a(i)$ is the average distance of point $i$ to other points in the same cluster
- $b(i)$ is the average distance of point $i$ to points in the nearest neighboring cluster

With `-1 as the poor clustering, and 1 as the well clustering`, this algorithm excels at general purpose, strong interpretability, and taking both elements of within-group and between-group conditions into account. However, this metric is sensitive to noise and outliers, sensitive to cluster shapes, and with high computational cost.
#### Calinski-Harabaz Index (Variance Ratio Criterion)

This method measures the ratio of between-cluster dispersion to within-cluster dispersion, which is given by$$\text{CH}=\frac{\text{Between-Cluster Dispersion}/(k-1)}{\text{Within-Cluster Dispersion}/(N-k)}$$
Where:
- $N$: Total number of points
- $k$: Number of clusters
- Between-Cluster Dispersion: Sum of squared distances between cluster centers and global center, i.e., $\sum^k_{i=1}n_i\|c_i-c\|^2$
- Within-Cluster Dispersion: Sum of squared distances between points and their cluster center, i.e., $\sum^k_{i=1}\sum_{x\in C_k}\|c_i-x\|^2$

With higher values indicating better clustering, this method assembles one-way ANOVA F-test statistics, which originally measures the F-score of equivalence of central tendency of multiple groups. Though effectively emphasizing dispersion and separation, this method is subject to sample sizes and number of clusters, e.g., not punishing over-clustering enough when $N$ goes up or $k$ goes to extremes.

#### Davies-Bouldin Index

This method measures the average similarity ratio of within-cluster scatter to between-cluster separation, which is given by$$\text{DB}=\frac{1}{k}\sum^k_{i=1}\max_{j\neq i}\left(\frac{S_i+S_j}{d(c_i,c_j)}\right)$$
where:
- $k$: Number of clusters
- $S_i$: Average distance of points in cluster $i$ to its center
- $d(c_i,c_j)$: Distance between centers of clusters $i$ and $j$.

With lower values indicating better clustering, this method focuses on cluster similarity, but might be sensitive to cluster shapes, scales, and distance metric, and also subject to extremes.

#### Dunn index

This method measures the ratio of the smallest inter-cluster distance to the largest intra-cluster diameter, which is given by$$\text{Dunn Index}=\frac{\min_{1\leq i\leq j\leq k}d(C_i,C_j)}{\max_{1\leq l\leq k}\text{diam}(C_l)}$$
Where:
- $d(C_i,C_j)$: Distance between clusters $C_i$ and $C_j$ (e.g., minimum or average distance)
- $\text{diam}(C_l)$: Diameter of cluster $C_l$ (e.g., maximum distance between points in the cluster)
- $k$: Number of clusters

With higher values indicating better clustering, this method encourages well-separated and compact clusters. However, it is sensitive to noise and outliers.


## Methods

It seems difficult to find a good clustering results by directly optimizing the above metrics, since:
- Non-differentiable: Most metrics are discrete or non-smooth, making gradient-based optimization infeasible.
- Combinatorial Complexity: Many require evaluation all possible cluster assignments, which is computationally intractable.
- Clustering Goal: Final natural groupings in data without predefined labels, unlike supervised learnings.

Therefore, we use heuristic algorithms that optimize proxy objectives instead, where metrics serve as evaluator and assess quality after clustering, not during optimization.

### (Hard) Prototype-based

`Definition`: Clustering methods that represent each cluster by a <font color=red>prototype (e.g., centroid, medoid)</font> summarizing the group.

`Key Idea`: Assign data points to the nearest prototype, optimizing a <font color=red>distance-based objective (e.g., sum of squares)</font>.

`Pros`: Simple, scalable, and effective for spherical clusters;
`Cons`: Sensitive to initialization and cluster shape.

[[K-Means and K-Medoids]]

### (Hard) Density-based

Though with improvements on `Prototype-based` Clustering algorithms, they obtain certain limitations like:
- Assume spherical, evenly sized clusters
- Require pre-specifying K
- Struggle with noise/outliers and non-convex shapes

Therefore, we try to relax the constraint by a `density perspective`. In short, we view clusters as regions of high point density separated by low-density areas, and in this view, we no need to assume cluster shape or specify $K$ in advance.

Key motivations are:
- Handle arbitrary cluster shapes (e.g., elongated, irregular)
- Identify noise/outliers naturally as low-density points
- Adapt to real-world data with varying densities

Thus, this kind of methods are more flexible and robust for complex datasets compared to prototype based methods.

[[Density-Based Spatial Clustering of Applications with Noise (DBSCAN)]]
[[Ordering Points To Identify the Clustering Structure (OPTICS)]]

### (Hard) Hierarchical

Hierarchical clustering is a clustering method that groups data by constructing a hierarchical structure of clusters. Its core lies in iteratively merging or splitting clusters to form a tree-like hierarchical relationship, ultimately presenting the clustering structure of data at different granularities. It is often used in fields that need to reveal the hierarchical relationships of data, such as biology (e.g., species classification), social sciences (e.g., group stratification), and data mining (e.g., hierarchical division of text topics).

The results of hierarchical clustering are usually displayed using a dendrogram:
- Horizontal axis: Data points or sub-clusters.
- Vertical axis: The distance at which clusters are merged (or split), reflecting the similarity between clusters (smaller distance indicates higher similarity).
- By "cutting" the dendrogram at a certain height, a specified number of clusters can be obtained.

`Advantages and Disadvantages`
- **Advantages**:
    - There is no need to pre-specify the number of clusters K; the clustering granularity can be flexibly selected through the dendrogram.
    - The dendrogram intuitively shows the hierarchical relationship of data, facilitating the understanding of the clustering structure.
- **Disadvantages**:
    - High computational complexity (especially for large datasets), resulting in high time costs.
    - Sensitivity to noise and outliers: Noise may cause irrelevant clusters to be merged, and outliers may delay reasonable splitting.

[[Agglomerative Hierarchical Clustering & Divisive Hierarchical Clustering]]
[[HDBSCAN]]


### Soft Clustering

Compared to hard clustering which determinedly assign every data point to one set, `Soft Clustering` assign every data point with `Memberships` (probability) with respect to various clusters. This aligns with the fact that real-world data is messy and often mixed.

[[Clustering-Gaussian Mixture Model]]
