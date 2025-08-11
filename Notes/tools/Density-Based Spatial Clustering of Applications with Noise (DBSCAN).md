# Intro: Key Concepts

Here we're to focus on distance between points, which are of geometry to some extent.

`ε-neighbors`: $N_\varepsilon(X_j)=\{x_i\in\mathcal{D}|\operatorname{dist}(x_i,x_j)\leq\varepsilon\}$

`Core points`: Points with at least $\text{minPts}$ neighbors within $\varepsilon$.

`Border points`: Points within $\varepsilon$ of a core point but not core points themselves.

`Noise points`: Points that are neither core nor border points, i.e., not reachable from any other point.

`Directly density-reachable`: $x_j$ is $\textcolor{green}{\text{directly density-reachable from }x_i}$ if $x_j\in N_\varepsilon(x_i)$ and $x_i$ is a core point

`Density-reachable`: $x_j$ is $\textcolor{green}{\text{density-reachable from a core point }x_i}$ if there is a chain $q_1,q_2,...q_n$, with $q_1=x_1,q_n=x_j$, $q_{k+1}$ is directly density-reachable from $q_k$

`Density-connected`: $x_i$ and $x_j$ are $\textcolor{green}{\text{density-connected}}$ if there is a $x_k$ such that $x_i$ and $x_j$ are both density-reachable from $x_k$

# Basic Idea

The basic idea is to try expanding the cluster based on density-reachability. Here, a cluster is a set of density-connected points which is maximal with respect to density-reachability:
- Connectivity: $\forall x_i,x_j\in C$, $x_i$ and $x_j$ are density-connected
- Maximality: $\forall x_i\in C$, if $x_j$ is density-reachable from $x_i\Rightarrow x_j\in C$

To achieve this, we would select an unvisited point and figure out:
- If it is a core point, create a new cluster by including all points density-reachable from it
- If it is not a core point, mark it as noisy (temporarily)
- After all iterations, assign each non-core point to a nearby cluster if the cluster is an $\varepsilon$ neighbor, otherwise assign it to noise

Then here we get the pseudocode.

# Pseudocode

![[Pasted image 20250802193531.png]]![[Pasted image 20250802193549.png]]![[Pasted image 20250802193607.png]]

If written in code, the structure should be like this:
**DBSCAN(DB, distFunc, eps, minPts)** {
    C := 0                                                  _/* Cluster counter */_
    **for each** point P **in** database DB {
        **if** label(P) ≠ undefined **then** **continue**               _/* Previously processed in inner loop */_
        Neighbors N := RangeQuery(DB, distFunc, P, eps)     _/* Find neighbors */_
        **if** |N| < minPts **then** {                              _/* Density check */_
            label(P) := Noise                               _/* Label as Noise */_
            **continue**
        }
        C := C + 1                                          _/* next cluster label */_
        label(P) := C                                       _/* Label initial point */_
        SeedSet S := N \ {P}                                _/* Neighbors to expand */_
        **for each** point Q **in** S {                             _/* Process every seed point Q */_
            **if** label(Q) = Noise **then** label(Q) := C          _/* Change Noise to border point */_
            **if** label(Q) ≠ undefined **then** **continue**           _/* Previously processed (e.g., border point) */_
            label(Q) := C                                   _/* Label neighbor */_
            Neighbors N := RangeQuery(DB, distFunc, Q, eps) _/* Find neighbors */_
            **if** |N| ≥ minPts **then** {                          _/* Density check (if Q is a core point) */_
                S := S ∪ N                                  _/* Add new neighbors to seed set */_
            }
        }
    }
}
**RangeQuery(DB, distFunc, Q, eps)** {
    Neighbors N := empty list
    **for each** point P **in** database DB {                      _/* Scan all points in the database */_
        **if** distFunc(Q, P) ≤ eps **then** {                     _/* Compute distance and check epsilon */_
            N := N ∪ {P}                                   _/* Add to result */_
        }
    }
    **return** N
}

# Optimization Criterion

DBSCAN optimizes the following loss function: For any possible clustering $C=\{C_1,...,C_l\}$ out of the set of all clusterings $\mathcal{C}$, it minimizes the number of clusters under the condition that every pair of points in a cluster is density-reachable, which corresponds to the original two properties "maximality" and "connectivity" of a cluster:$$\min_{C\in\mathcal{C},d_{db}(p,q)\leq\varepsilon,\forall p,q\in C_i,\forall C_i\in C}|C|$$where $d_{db}(p,q)$ gives the smallest $\varepsilon$ such that two points p and q are density-connected.

# Properties, Pros, and Cons

For properties, we have:
1. Core points are deterministically assigned
	- Two core points $p$ and $q$ belong to the same cluster if there exists a chain of core points
	- This relation $p\sim q$ is an equivalence relation because it is reflexive, symmetric, and transitive
	- Then we obtain a partition induced by this equivalence relation
2. Noise points are deterministically assigned
	- Noise point is not a core point and it is not within the ε -neighborhood of any core point
3. Border points are non-deterministically assigned
	- A border point is not a core point but lies within the ε -neighborhood of at least one core point
	- The assignment depends on the order in which core points are processed

For Pros, we have
- Does not require the number of clusters to be specified
- Can identify clusters of arbitrary shape
- Handles noise effectively by identifying outliers

For Cons, we have:
- Not entirely deterministic
- Performance degrades with high-dimensional data: **curse of dimensionality** (data points appear very sparse since every 2 randomly selected vectors appear almost orthogonal)
- Sensitive to the choice of $\varepsilon$ and $\text{minPts}$
	- $\text{minPts}=2\cdot \text{dim}$ is a good choice, but may need to try a larger value for a larger dataset.
	- $\varepsilon$: plotting the distance to the $k=\text{minPts}-1$ nearest neighbor ordered from the largest to the smallest value. And like the elbow method, find one "elbow" in the plot where an increase in $\varepsilon$ leads to decrease in $k$.
- Struggles with clusters of varying densities. When some clusters appear very sparse while others do not, the algorithm fails, which we would improve in [[Ordering Points To Identify the Clustering Structure (OPTICS)]].