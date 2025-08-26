
# Motivation

Think about this dataset:![[Pasted image 20250802202119.png]]From this dataset, we might see that A,C are sparse, while $B_1$, $B_2$, and $B_3$ are dense. However, by [[Density-Based Spatial Clustering of Applications with Noise (DBSCAN)]], we could only get $\{A,B,C\},\{B_1,B_2,B_3\},...$ instead of $\{A,B_1,B_2,B_3,C\}$. Therefore, we wish for an algorithm that:
- Relax $\varepsilon$ in DBSCAN from a value to a value range
- Larger $\varepsilon$ for sparser regions and smaller $\varepsilon$ for denser areas

# Key Concepts

For hyperparameters, we have
- $\varepsilon$: **maximum radius**, which could be set as $\infty$. A smaller value can make the algorithm run faster
- $\text{minPts}$: minimum number of points to form a cluster

Then, we have key concepts to achieve adaptive $\varepsilon$
- **Core Distance**$$\operatorname{cd}(p)=\begin{cases}\text{Undefined},\quad\quad\ \ |N_\varepsilon(p)|<\text{minPts}\\d(p,N_\varepsilon^{\text{minPts}}(p)),\ |N_\varepsilon(p)|\geq\text{minPts}\end{cases}$$Which is the minimum distance to make a point $p$ to be a core point. Such $\varepsilon$ could be viewed as "Local density".
- **Reachability Distance**$$\operatorname{rd}(q,p)=\begin{cases}\text{Undefined}, \quad\quad\quad\quad |N_\varepsilon(p)|<\text{minPts}\\\max\{\text{cd}(p),d(p,q)\},\ |N_\varepsilon(p)|\geq\text{minPts}\end{cases}$$Which is the minimum distance to make $p$ to be a core point and $q$ is directly reachable from $p$. Further, we define $r_j:=\min_{i\text{ is processed}}\text{rd}(j,i)$.

# Basic Idea

The basic approach of OPTICS is similar to DBSCAN, but instead of `maintaining known, but so far unprocessed cluster members` in a set, they are maintained in a priority queue (i.e., indexed heap).

# Pseudocode



![[Pasted image 20250802203956.png]]![[Pasted image 20250802204717.png]]

code version:
**function** OPTICS(DB, ε, MinPts) **is**
    **for each** point p of DB **do**
        p.reachability-distance = UNDEFINED                   _/* Initialization */_
    **for each** unprocessed point p of DB **do**
        N = getNeighbors(p, ε)                                  _/* Get points of neighbor */_
        mark p as processed
        output p to the ordered list                               _/* Order of processing */_
        **if** core-distance(p, ε, MinPts) != UNDEFINED **then**
            Seeds = empty priority queue
            update(N, p, Seeds, ε, MinPts)
            **for each** next q in Seeds **do**
                N' = getNeighbors(q, ε)
                mark q as processed
                output q to the ordered list
                **if** core-distance(q, ε, MinPts) != UNDEFINED **do**
                    update(N', q, Seeds, ε, MinPts)
**function** update(N, p, Seeds, ε, MinPts) **is**
    coredist = core-distance(p, ε, MinPts)
    **for each** o in N
        **if** o is not processed **then**
            new-reach-dist = max(coredist, dist(p,o))
            **if** o.reachability-distance == UNDEFINED **then** // o is not in Seeds
                o.reachability-distance = new-reach-dist
                Seeds.insert(o, new-reach-dist)
            **else**               // o in Seeds, check for improvement
                **if** new-reach-dist < o.reachability-distance **then**
                    o.reachability-distance = new-reach-dist
                    Seeds.move-up(o, new-reach-dist)

![[Pasted image 20250802214511.png]]![[Pasted image 20250802214854.png]]

# Reachability Plots and Clustering method

OPTICS generates a reachability plot that reveals the hierarchical relationships between clusters, with x-axis as order of points, and y-axis as reachability distances. Then, Each cluster is a 'U' shape valley in this plot.

Then, we conduct clustering based on the plot visualized by identifying valleys (low reachability distances), which are:
- Xi Method: Automatically detect clusters by steepness in the reachability plot (parameter $\xi$).
- Cut-off Threshold: Manually set a reachability distance threshold to separate clusters.

# Pros and Cons

Then, we have pros and cons.
1. Pros
	- Flexible: Detects clusters with varying densities, unlike DBSCAN.
	- Hierarchical: Produces a reachability plot for multiple cluster resolutions.
	- Robust: Less sensitive to parameter choices than DBSCAN.
	- Noise Handling: Effectively identifies outliers.
2. Cons
	- Computational Cost: Slower than DBSCAN for large datasets $(O(n^{2})$ without indexing).
	- Parameter Tuning: Requires setting ε and MinPts, which can affect results.
	- Interpretation: Reachability plots may need expertise to analyze.