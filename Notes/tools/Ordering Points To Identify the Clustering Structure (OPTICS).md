
**Document Class** #Clustering  #DSAA2011 

**Table of Contents:**
- [[#Motivation|Motivation]]
- [[#Key Concepts and Basic Idea|Key Concepts and Basic Idea]]
- [[#Pseudocode|Pseudocode]]
- [[#Example (Hands on Practice)|Example (Hands on Practice)]]
	- [[#Example (Hands on Practice)#First Iteration|First Iteration]]
	- [[#Example (Hands on Practice)#Second Iteration|Second Iteration]]
	- [[#Example (Hands on Practice)#Further Iterations|Further Iterations]]
	- [[#Example (Hands on Practice)#Result|Result]]
- [[#Reachability Plots and Clustering method|Reachability Plots and Clustering method]]
- [[#Pros and Cons|Pros and Cons]]

**Improved [[Density-Based Spatial Clustering of Applications with Noise (DBSCAN)|DBSCAN]] with altered ε.**

---
## Motivation

Think about this dataset:![[Pasted image 20250802202119.png]]From this dataset, we might see that A,C are sparse, while $B_1$, $B_2$, and $B_3$ are dense. However, by [[Density-Based Spatial Clustering of Applications with Noise (DBSCAN)]], we could only get $\{A,B,C\},\{B_1,B_2,B_3\},...$ instead of $\{A,B_1,B_2,B_3,C\}$. Therefore, we wish for an algorithm that:
- Relax $\varepsilon$ in DBSCAN from a value to a value range
- Larger $\varepsilon$ for sparser regions and smaller $\varepsilon$ for denser areas

## Key Concepts and Basic Idea

For hyperparameters, we have
- $\varepsilon$: **maximum radius**, which could be set as $\infty$. A smaller value can make the algorithm run faster
- $\text{minPts}$: minimum number of points to form a cluster

Then, we have key concepts to achieve adaptive $\varepsilon$
- **Core Distance**$$\operatorname{cd}(p)=\begin{cases}\text{Undefined},\quad\quad\ \ |N_\varepsilon(p)|<\text{minPts}\\d(p,N_\varepsilon^{\text{minPts}}(p)),\ |N_\varepsilon(p)|\geq\text{minPts}\end{cases}$$Which is the minimum distance to make a point $p$ to be a core point (a concept in [[Density-Based Spatial Clustering of Applications with Noise (DBSCAN)#Intro Key Concepts|DBSCAN Concepts]]). Such $\varepsilon$ could be viewed as "Local density". In other words, $\text{cd}(p)$ is the $minPts^\text{th}$ distance within the $\varepsilon$ neighborhood.
- **Reachability Distance**$$\operatorname{rd}(q,p)=\begin{cases}\text{Undefined}, \quad\quad\quad\quad |N_\varepsilon(p)|<\text{minPts}\\\max\{\text{cd}(p),d(p,q)\},\ |N_\varepsilon(p)|\geq\text{minPts}\end{cases}$$Which is the minimum distance to make $p$ to be a core point and $q$ is directly reachable from $p$. Further, we define $r_j:=\min_{i\text{ is processed}}\text{rd}(j,i)$.

The basic approach of OPTICS is similar to DBSCAN, but instead of `maintaining known, but so far unprocessed cluster members` in a set, they are maintained in a priority queue (i.e., indexed heap).

## Pseudocode

`Algorithm: OPTICS`
**Require:** $\varepsilon, minPts$
- $k\gets1, I\gets\{1,2,\cdots,N\}, r_i\gets UND$
- **while** $I$ is unprocessed **then**
	- Get an element $i$ from $I$, and let $I\gets I\setminus\{i\}$
	- **if** $|N_\varepsilon(j)|\geq minPts$ **then**
		- Seeds $=$ empty priority queue
		- $\text{Update}\left(N_\varepsilon(i), i, \text{Seeds}, \varepsilon, minPts\right)$
		- **while** Seeds is not empty **do**
			- Remove the ==***top element***== $j$ from Seeds; *# Min reachability dist*
			- Mark $j$ as processed, $p_k\gets j, k\gets k+1$
			- **if** $|N_\varepsilon(j)|\geq minPts$ **then** $\text{Update}(N_\varepsilon(j),j,\text{Seeds},\varepsilon, minPts)$
		- **end while**
	- **end if**

`Algorithm: Update Priority Queue`
**Require:** $N_\varepsilon(i),i,\text{Seeds},\varepsilon, minPts$
- $c_i=\text{core distance}(i,\varepsilon,minPts)$
- **for** each unprocessed element $j$ in $N_\varepsilon(i)$ **do**
	- $r_j^\text{new}=\max(c_i,d(i,j))$
	- **if** $r_j=UND\ \text{OR}\ r_j^\text{new}<r_j$ **then** *# Relax $r_j$*
		- $r_j\gets r_j^\text{new}$
		- Seeds.Insert$(j,r_j^\text{new})$ *# insert or update point j with priority $r_j^\text{new}$*
	- **end if**
- **end for**

***Also, a `code` version in wikipedia:***

**function** OPTICS(DB, ε, MinPts) **is**
- **for each** point p of DB **do**
	- p.reachability-distance = UNDEFINED                   _# Initialization_
- **for each** unprocessed point p of DB **do**
	- N = getNeighbors(p, ε)                                  _# Get points of neighbor_
	- mark p as processed
	- output p to the ordered list                               _# Order of processing_
	- **if** core-distance(p, ε, MinPts) != UNDEFINED **then**
		- Seeds = empty priority queue
		- update(N, p, Seeds, ε, MinPts)
		- **for each** next q in Seeds **do**
			- N' = getNeighbors(q, ε)
			- mark q as processed
			- output q to the ordered list
			- **if** core-distance(q, ε, MinPts) != UNDEFINED **do**
				- update(N', q, Seeds, ε, MinPts)

**function** update(N, p, Seeds, ε, MinPts) **is**
- coredist = core-distance(p, ε, MinPts)
- **for each** o in N
	- **if** o is not processed **then**
		- new-reach-dist = max(coredist, dist(p,o))
		- **if** o.reachability-distance == UNDEFINED **then** _# o is not in Seeds_
			- o.reachability-distance = new-reach-dist
			- Seeds.insert(o, new-reach-dist)
		- **else**               // o in Seeds, check for improvement
			- **if** new-reach-dist < o.reachability-distance **then**
				- o.reachability-distance = new-reach-dist
				- Seeds.move-up(o, new-reach-dist)


## Example (Hands on Practice)

Here is one provided example with $minPts=3,\varepsilon=4$. You might start it with **L**, and try to build up on your own.

*Note: it's a lab exercise I've viewed on Youtube, but unfortunately the video resource was soon unavailable. Please inform anything about the copyrights.*

![[Pasted image 20250802214511.png]]

### First Iteration

- For the first Iteration, we search **L** and set $\text{rd}(L)=\infty, \text{cd}(L)=4$.
- Then search its $\varepsilon$ neighbors, which are $\textcolor{red}{(K,4),(M,4),(O,4)}$ based on the core distance value. Please notice that they are ordered in a heap structure by reachability distance.
- Since **L** is marked processed, we put it to the queue of the final output with $(L,\infty)$.

### Second Iteration

- For the second Iteration, we drop $(K,4)$ from top of the heap structure and search along $K$.
- Here we figured out the $\varepsilon$ neighbor of $\textcolor{green}{(J,4),(H,4),(I,4),(G,4)}$ based on $\text{cd}(K)=4$.
- Then put or update the searched neighbor into the heap, and we get the heap as $\textcolor{red}{(M,4),(O,4),(J,4),(H,4),(I,4),(G,4)}$
- $K$ is marked processed and updated to the final output with $(K,4)$.

### Further Iterations

You're welcome to check a few more iterations on OPTICS based on above instructions.

**Iteration 3**
	Search & Final Output Sequence: $(M,4)$
	Updated/Added Nodes: $\textcolor{green}{(P,2),(O,2),(N,2),(R,2),(Q,3),(S,3),(N,4),(T,4)}$
	Updated Heap: $\textcolor{red}{(P,2),(O,2),(N,2),(R,2),(Q,3),(S,3),(N,4),(T,4),(J,4),(H,4),(I,4),(G,4)}$
	
**Iteration 4**
	Search & Final Output Sequence: $(P,2)$
	Updated/Added Nodes: $\textcolor{green}{(O,2),(N,2),(R,2),(S,2),(Q,3),(T,3)}$
	Updated Heap: $\textcolor{red}{(O,2),(N,2),(R,2),(S,2),(Q,3),(T,3),(J,4),(H,4),(I,4),(G,4)}$
	
**Iteration 5**
	Search & Final Output Sequence: $(O,2)$
	Updated/Added Nodes: $\textcolor{green}{(R,1),(N,1),(Q,2),(S,2),(T,2)}$
	Updated Heap: $\textcolor{red}{(R,1),(N,1),(Q,2),(S,2),(T,2),(J,4),(H,4),(I,4),(G,4)}$
	

### Result
A plausible result is given below. It's the final output sequence we're dealing with. Here, <font color=red>a color represents a cluster</font>, which is detected by human or other algorithms.
![[Pasted image 20250802214854.png]]

## Reachability Plots and Clustering method

OPTICS generates a reachability plot that reveals the hierarchical relationships between clusters, with x-axis as order of points, and y-axis as reachability distances. Then, Each cluster is a 'U' shape valley in this plot.

Then, we conduct clustering based on the plot visualized by identifying valleys (low reachability distances), which are:
- Xi Method: Automatically detect clusters by steepness in the reachability plot (parameter $\xi$).
- Cut-off Threshold: Manually set a reachability distance threshold to separate clusters.

## Pros and Cons

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
