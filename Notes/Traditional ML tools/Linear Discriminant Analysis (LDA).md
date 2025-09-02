
Assume we have dataset $X$ and the associated class labelsd $Y$, then how can we leverage the information from the class labels for better dimensionality reduction? This sets up a different goal: Projecting data to lower dimensions while maximizing class differences.

# Mathematical Inference

We take separation for two classes as the example here. Then after projection $v$ to one dimensional space, we have$$\mu_1=\frac{1}{n_1}\sum_{i\in C_1}a_i=v^T\sum_{i\in C_1}\frac{1}{n_1}x_i:=v^Tm_1,\mu_2=v^Tm_2,s_1^2=\sum_{i\in C_1}(a_i-\mu_1)^2,s_2^2=\sum_{i\in C_2}(a_i-\mu_2)^2$$to introduce a goal considering both the mean and the sample standard deviation$$\max_{v:\|v\|=1}\frac{(\mu_1-\mu_2)^2}{s_1^2+s_2^2}$$For further analysis.

First, we consider the numerator$$(\mu_1-\mu_2)^2=(v^T(m_1-m_2))^2=v^T(m_1-m_2)(m_1-m_2)^Tv:=v^T\cdot S_b\cdot v$$where $S_b\in \mathbb{R}^{p\times p}$ is the between-class scatter matrix. Also, for the denominator$$\begin{aligned}s_1^2+s_2^2&=\sum_{i\in C_1}v^T(x_i-m_1)(x_i-m_1)^Tv+\sum_{i\in C_2}v^T(x_i-m_2)(x_i-m_2)^Tv\\&=v^T\left(\sum_{i\in C_1}(x_i-m_1)(x_i-m_1)^T+\sum_{i\in C_2}(x_i-m_2)(x_i-m_2)^T\right)v\\:&=v^TS_wv\end{aligned}$$where $S_w\in\mathbb{R}^{p\times p}$ is the within-class scatter matrix. Suppose it is invertible, and we will have the optimal vector $v^*$ should be an eigenvector of $S_w^{-1}S_b$ with corresponding largest eigenvalue since $\frac{v^TS_bv}{v^TS_wv}=\frac{u^TS_w^{-1/2}S_bS_w^{-1/2}u}{u^Tu}$. In fact, $\lambda_1v_1=S_w^{-1}S_bv_1=S_w^{-1}(m_1-m_2)(m_1-m_2)^Tv_1\propto S_w^{-1}(m_1-m_2)$, indicating the best projection direction is corresponded to the mean diference, which helps separating two classes. (Note that $0<\text{Rank}(S_w^{-1}S_b)\leq\text{Rank}(S_b)=1$, so there is no second eigenvector in this example, or binary classification.)

For generalization to $C\geq3$ classes, we could measure the separation by$$\begin{aligned}S_b&=\sum_{j=1}^Cn_j(m_j-m)(m_j-m)^T\\S_w&=\sum^C_{j=1}\sum_{i\in C_j}(x_i-m_j)(x_i-m_j)^T\end{aligned}$$which is a generalization parameter setting which get the similar and actually same objective function for optimization. This leads us to the steps or actual algorithm.

# Steps

- Given a dataset $D \in \mathbb{R}^{n ×p}$ and the associated class labels $Y$ of $C$ classes
- Compute $S_{w}=\sum_{j=1}^{C} \sum_{i \in C_{j}}(x-m_{j})(x-m_{j})^{\top}$ and $S_{b}=\sum_{j=1}^{C} n_{j}(m_{j}-m)(m_{j}-m)^{\top}$ where $C_{j}$ is the set of index belong to class j , $n_{j}=|C_{j}|$ is the number of samples belong to class j , $m_{j}=\frac{1}{n_{j}} \sum_{i \in C_{j}} x_{j}$ is the mean of samples in class j , and $m=\frac{1}{n}\sum^n_{i=1}x_i$ is the mean of all samples
- Solve the generalized eigenvalue problem $S_bv=\lambda S_wv$ to find all eigen values and their associated eigenvectors $V=\{v_1,\cdots,v_k\}$
- Project $Y = XV\in\mathbb{R}^{n\times k}$

# Pros & Cons

1. Pros
	- Effectively leverage label information to maximize class separability
	- Effective for classification tasks
2. Cons
	- At most $C-1$ feature projections
	- Assumes linear class boundaries
	- Assumes Gaussian-distributed data with equal covariance matrix
	- It assumes that different classes are identifiable through the mean

# Extension: [[Quadratic Discriminant Analysis (QDA)]]
