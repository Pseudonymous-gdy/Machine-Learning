
**Document Class** #DSAA2011  #Supervised-Learning  #Classification

**Table of Contents:**
- [[#Intro: Definitions and Motivations|Intro: Definitions and Motivations]]
- [[#Formula & Explanations|Formula & Explanations]]
- [[#Soft Margin|Soft Margin]]
- [[#Model Evaluation|Model Evaluation]]
- [[#Advanced Techniques|Advanced Techniques]]
	- [[#Advanced Techniques#Solution of Dual Form through SMO (Sequential Minimal Optimization)|Solution of Dual Form through SMO (Sequential Minimal Optimization)]]
		- [[#Solution of Dual Form through SMO (Sequential Minimal Optimization)#Dual Form|Dual Form]]
		- [[#Solution of Dual Form through SMO (Sequential Minimal Optimization)#SMO Algorithm|SMO Algorithm]]
			- [[#SMO Algorithm#Core Idea|Core Idea]]
			- [[#SMO Algorithm#Key Concepts|Key Concepts]]
			- [[#SMO Algorithm#SMO Algorithm Steps|SMO Algorithm Steps]]
			- [[#SMO Algorithm#Advantages|Advantages]]
			- [[#SMO Algorithm#Practical Considerations|Practical Considerations]]
			- [[#SMO Algorithm#Applications|Applications]]
			- [[#SMO Algorithm#Pseudo-code|Pseudo-code]]
	- [[#Advanced Techniques#Kernel Trick|Kernel Trick]]
- [[#Codes|Codes]]
- [[#Complement|Complement]]

**The first nearly 100% mathematically-based classification model.**

---
## Intro: Definitions and Motivations

`Margin of Classifier`:

Since given a dataset $\mathcal{D}=\{(\mathbf{x}_t,y_t)\in\mathbb{R}^d\times\{-1,+1\}\}$, we can find some classifier $f_{\theta}=\operatorname{sign}(\mathbf{\theta}^T\mathbf{x})$ indicates that $y\mathbf{\theta}^T\mathbf{x}>0$, we define the `margin` of classifier $f_{\theta}=\operatorname{sign}(\mathbf{\theta}^T\mathbf{x})$ on sample $(\mathbf{x},y)$ as $y\mathbf{\theta}^T\mathbf{x}$ or $y\left\langle\mathbf{\theta},\mathbf{x}\right\rangle$.

This is defined based on the hyperplanes of an Euclidean Space $\mathbb{R}^d$, where a hyperplane necessarily has an orthogonal vector. However, this could only define the 'direction' of the hyperplane, while in reality we need a scalar offset $\theta_0$, which is derived from $\mathbf{\theta}^T(\mathbf{x}+\mathbf{v})=\mathbf{\theta}^T\mathbf{x}+\mathbf{\theta}^T\mathbf{v}=\mathbf{\theta}^T\mathbf{x}+\theta_0$. <font color=red>Then this is the definition of Affinely Separable, compared with the linearly separable below</font>.

If $y\mathbf{\theta}^T\mathbf{x}>0$, we might conclude that the data points of two classes could be separated by a straight line, and in high-dimensional view, a hyperplane.

`Linearly Separable`

The dataset $\mathcal{D}$ is `linearly separable` if there exists some $\mathbf{\theta}$ such that
$$
y_t\mathbf{\theta}^T\mathbf{x}_t>0,\forall t=1,2,...,n$$

Until now, we could use a hyperplane to divide two categories. However, it might be more important to determine one final hyperplane that is robust for generalized datasets. In this circumstance, the hyperplane should allow as much error as possible when predicting. This changes to discussions of distances, which is based on the following definition.

`γ-Linearly Separable`

The dataset is $\gamma$-linearly separable for some $\gamma>0$ if there exists some $\theta$ such that
$$y\mathbf{\theta}^T\mathbf{x}>\gamma
$$

Therefore, there is some hyperplane that strictly separates the data into positive and negative samples with:
- $\mathbf{\theta}$ has positive margin $y_t\mathbf{\theta}^T\mathbf{x}_t>0$ for all t.
- Minimum margin $\gamma=\min_{1\leq t\leq n}y_t\mathbf{\theta}^T\mathbf{x}_t$.

Notice that Geometric distance between the data points and the hyperplane is related to the formula above, with the basic knowledge of inner product:

$$
\gamma_{t_{Geom}}:=|\mathbf{x}_t|\cdot\cos\omega=\frac{\left\langle\mathbf{\theta},\mathbf{x}_t\right\rangle+\theta_0}{\|\mathbf{\theta}\|}
$$

Here, $\gamma_{t_{Geom}}$ is exactly the distance. So we might take steps to learn about optimizing all $\gamma_{t_{Geom}}$ through maximizing the minimum distance.

## Formula & Explanations

We therefore define
$$
\gamma_{_{Geom}}:=\frac{\gamma}{\|\mathbf{\theta}\|}=\frac{\min_{1\leq t\leq n}y_t\left(\mathbf{x}_t^T\mathbf{\theta}+\theta_0\right)}{\|\mathbf{\theta}\|}
$$
And conclude the optimization form of
$$\begin{aligned}
\max \quad & \frac{\min_{1\leq t\leq n}y_t\left(\mathbf{x}_t^T\mathbf{\theta}+\theta_0\right)}{\|\mathbf{\theta}\|} = \frac{\gamma}{\|\mathbf{\theta}\|} \\
\text{s.t.} \quad &y_{i}\left(\mathbf{x}_i^T\mathbf{\theta}+\theta_0\right)\geq \gamma,\forall i=1,2,...,n
\end{aligned}$$
Which is
$$\begin{aligned}
\min \quad & \frac{\|\mathbf{\theta}\|}{\gamma}\\
\text{s.t.} \quad &y_{i}\left(\mathbf{x}_i^T\mathbf{\theta}+\theta_0\right)\geq \gamma,\forall i=1,2,...,n
\end{aligned}$$

And
$$\begin{aligned}
\min \quad & \|\mathbf{\theta}\|\\
\text{s.t.} \quad &y_{i}\left(\mathbf{x}_i^T\mathbf{\theta}+\theta_0\right)\geq 1,\forall i=1,2,...,n
\end{aligned}$$

For further convenience (transformation to dual form), we have:

$$\begin{aligned}
\min \quad & \frac{1}{2}\|\mathbf{\theta}\|^2\\
\text{s.t.} \quad &y_{i}\left(\mathbf{x}_i^T\mathbf{\theta}+\theta_0\right)\geq 1,\forall i=1,2,...,n
\end{aligned}$$

The solution is unique, for the objective function is convex and the constraint function is linear. To be detailed, if $\mathbf{\theta}_1$ and $\mathbf{\theta}_2$ are two distinct minimal solution, then $\lambda\mathbf{\theta}_1+(1-\lambda)\mathbf{\theta}_2$ satisfies the constraint while reaches a possibly lower value with equivalence guaranteed only if $\mathbf{\theta}_1 \mathop{//} \mathbf{\theta}_2$.

After optimization, we would find out that there should be certain $i$s such that $y_{i}\left(\mathbf{x}_i^T\mathbf{\theta}+\theta_0\right)=0$, which we would call as `support vectors`, indicating that samples are exactly on the margin.

## Soft Margin

If the data is not linear/affinely-separable, there are usually two ways to settle the problem:
- map to higher dimension (i.e., infinite dimensions). This would be included in [[#Advanced Techniques#Kernel Trick]].
- allow certain misclassified sample, to make the model more robust.

Therefore, we introduce slack variables, allowing that $y_{i}\left(\mathbf{x}_i^T\mathbf{\theta}+\theta_0\right)\geq 1-\xi_i,\forall i=1,2,...,n$. Thus the optimization of `Soft Margin` becomes:
$$\begin{aligned}
\min \quad & \frac{1}{2}\|\mathbf{\theta}\|^2 + C\sum^n_{i=1}\xi_i\\
\text{s.t.} \quad &y_{i}\left(\mathbf{x}_i^T\mathbf{\theta}+\theta_0\right)\geq 1-\xi_i,\forall i=1,2,...,n\\
&\xi_i\geq0,\forall i=1,2,...,n
\end{aligned}$$
Hence another form is
$$\min\quad\frac{1}{2}\|\mathbf{\theta}\|^2+C\sum^n_{t=1}\mathop{ReLU}(1-y_t(\mathbf{x}_t^T\mathbf{\theta}+\theta_0))$$

## Model Evaluation

We adopt the evaluation method as:
$$
\frac{\#\text{incorrect prediction}}{\#\text{all prediction}}=\frac{\sum^n_{i=1}\operatorname{Loss}\Bigl(y_t,f\left(\mathbf{x}_t,(\mathbf{\theta},\theta_0)\right)\Bigr)}{n}=\frac{\sum^n_{i=1}\mathbb{1}\{y_t,f\left(\mathbf{x}_t,(\mathbf{\theta},\theta_0)\right)\}}{n}
$$

Also, to assess the `robustness` of the SVM, we adopt `Leave-one-out cross-validation` error (LOOCV):

$$
\text{LOOCV}:=\frac{1}{n}\sum^n_{t=1}\mathbb{1}\{y_t,f\left(\mathbf{x}_t,(\mathbf{\theta}^{-t},\theta_0^{-t})\right)\}
$$

Where $(\mathbf{\theta}^{-t},\theta_0^{-t})$ is learnt from the dataset with $t^\text{th}$ sample left out, i.e., $\mathcal{D}^{-t}=\mathcal{D}\setminus\{\mathbf{x}_t,y_t\}$.

Then for hard margin, we have $\text{LOOCV}=\frac{N}{n}$, where $N$ is the number of support vectors in the dataset. (Reasoned by $$\text{\textcolor{blue}{If }}\textcolor{blue}{\mathbf{x}_t}\text{\textcolor{blue}{ is not a support vector, then it cannot influence the solution of the primal SVM}}$$which we will mention a sketch in [[#^eedf00|Complement No.1]]).

## Advanced Techniques
### Solution of Dual Form through SMO (Sequential Minimal Optimization)
#### Dual Form

We use the Lagrangian Multipliers $\alpha_n,\gamma_n\geq0$ to the primal form, and get:
$$\begin{aligned}
\mathcal{L}(\mathbf{\theta},\theta_0,\mathbf{\xi},\mathbf{\alpha},\mathbf{\gamma})&=\frac{1}{2}\|\mathbf{\theta}\|^2 + C\sum^n_{i=1}\xi_i\\
&\quad\textcolor{blue}{-\sum^n_{t=1}\alpha_t(y_t\left(\left\langle\mathbf{\theta},\mathbf{x}_t\right\rangle+\theta_0\right)-1+\xi_t)-\sum^n_{t=1}\gamma_t\xi_t}
\end{aligned}$$
By differentiating the formula, we obtain
$$\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{\theta}}&=\mathbf{\theta}^T-\sum^n_{t=1}\alpha_ty_t\mathbf{x}_t^T\\
\frac{\partial \mathcal{L}}{\partial \theta_0}&=-\sum^n_{t=1}\alpha_ty_t\\
\frac{\partial \mathcal{L}}{\partial \xi_t}&=C-\alpha_t-\gamma_t
\end{aligned}$$
Where we get $\mathbf{\theta}=\sum^n_{t=1}\alpha_ty_t\mathbf{x}_t$. Then by substituting the expression, we obtain the dual form
$$\begin{aligned}
\mathcal{Dual}(\xi,\alpha,\gamma)&=\frac{1}{2}\sum^n_{t=1}\sum^n_{s=1}y_ty_s\alpha_t\alpha_s\left\langle\mathbf{x}_t,\mathbf{x}_s\right\rangle-\sum^n_{t=1}y_t\alpha_t\left\langle\sum^n_{s=1}y_s\alpha_s\mathbf{x}_s,\mathbf{x}_t\right\rangle\\
&\quad+C\sum^n_{i=1}\xi_i-\theta_0\sum^n_{t=1}y_t\alpha_t+\sum^n_{t=1}\alpha_t-\sum^n_{t=1}\alpha_t\xi_t-\sum^n_{t=1}\gamma_t\xi_t\\
&=-\frac{1}{2}\sum^n_{t=1}\sum^n_{s=1}y_ty_s\alpha_t\alpha_s\left\langle\mathbf{x}_t,\mathbf{x}_s\right\rangle+\sum^n_{t=1}\alpha_t+\sum^n_{t=1}(C-\alpha_i-\gamma_i)\xi_i
\end{aligned}$$
Then we conclude a minimization of the negative dual problem, such that we end up with the dual SVM
$$\begin{aligned}
\min_\alpha\quad & \frac{1}{2}\sum^n_{t=1}\sum^n_{s=1}y_ty_s\alpha_t\alpha_s\left\langle\mathbf{x}_t,\mathbf{x}_s\right\rangle-\sum^n_{t=1}\alpha_i\\
\text{s.t.}\quad & \sum^n_{t=1}y_t\alpha_t=0\\
&0\leq\alpha_i\leq C,\forall i=1,2,...,n
\end{aligned}$$

Notice the above ones are built on `Karush-Kuhn-Tucker` Condition.

Also, for hard margin, the upper constraint on $\alpha_i$ is eliminated since no slack variables are introduced; for kernel functions, the inner product is turned into kernel functions, and the results are turned into:
$$\begin{aligned}
\min_\alpha\quad & \frac{1}{2}\sum^n_{t=1}\sum^n_{s=1}y_ty_s\alpha_t\alpha_sK\Bigl(\mathbf{x}_t,\mathbf{x}_s\Bigr)-\sum^n_{t=1}\alpha_i\\
\text{s.t.}\quad & \sum^n_{t=1}y_t\alpha_t=0\\
&0\leq\alpha_i\leq C,\forall i=1,2,...,n
\end{aligned}$$

Then we introduce the algorithm SMO.
#### SMO Algorithm
The **Sequential Minimal Optimization (SMO)** algorithm, introduced by John Platt in 1998, is an efficient method for training **Support Vector Machines (SVMs)**. It solves the dual optimization problem of SVMs by breaking it into smaller subproblems, enabling fast convergence and scalability. Below is a structured overview:

##### Core Idea
SMO decomposes the large quadratic programming (QP) problem of SVM training into the **smallest possible subproblems**: optimizing two Lagrange multipliers (αᵢ, αⱼ) per iteration. This avoids complex numerical QP solvers and leverages analytical updates. *A basic explanation provided by [Youtubers](https://www.youtube.com/watch?v=Mfp7HQKLSAo) points out we probably need to optimize the function on one variable under the range condition with changes to another variable that is respectively changed.*

##### Key Concepts
1. **SVM Dual Problem**:
   - **Objective**: Maximize  
    $$ W(\alpha) = \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i,j=1}^m \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) $$
   - **Constraints**:  
     $0 \leq \alpha_i \leq C$ (box constraints),  
     $\sum_{i=1}^m \alpha_i y_i = 0$ (linear constraint),  
     where $y_i \in \{-1, +1\}$ are labels, $K$ is a kernel, and $C$ is the regularization parameter.

2. **Why Two Multipliers?**  
   The linear constraint $\sum \alpha_i y_i = 0$ requires adjusting at least two α’s simultaneously to maintain feasibility.

##### SMO Algorithm Steps
1. **Initialization**:  
   Set all αᵢ = 0 and bias $\theta_0=0$.

2. **Heuristic Selection of Multipliers**:  
   - **First αᵢ**: Choose an αᵢ violating the **Karush-Kuhn-Tucker (KKT)** conditions:  
     $$ \alpha_i = 0 \implies y_i f(\mathbf{x}_i) \geq 1, $$
     $$ \alpha_i = C \implies y_i f(\mathbf{x}_i) \leq 1, $$
     $$ 0 < \alpha_i < C \implies y_i f(\mathbf{x}_i) = 1. $$
   - **Second αⱼ**: Select αⱼ maximizing the step size (via $|E_i - E_j|$), where $E_i = f(\mathbf{x}_i) - y_i$ is the prediction error.

2. **Analytical Update**: 
   - Compute bounds $L$ and $H$ for αⱼ: 
     $$ L = \max(0, \alpha_j - \alpha_i), \ H = \min(C, C + \alpha_j - \alpha_i) \ \text{if} \ y_i \neq y_j, $$$$ L = \max(0, \alpha_i + \alpha_j - C), \ H = \min(C, \alpha_i + \alpha_j) \ \text{if} \ y_i = y_j. $$
   - Update αⱼ:$$ \alpha_j^{\text{new}} = \alpha_j^{\text{old}} - \frac{y_j (E_i - E_j)}{\eta}, $$
     where $\eta = 2K(\mathbf{x}_i, \mathbf{x}_j) - K(\mathbf{x}_i, \mathbf{x}_i) - K(\mathbf{x}_j, \mathbf{x}_j)$ (curvature). 
     Clip to $[L, H]$: $\alpha_j^{\text{new}} \leftarrow \text{clip}(\alpha_j^{\text{new}}, L, H)$.
   - Update αᵢ:
     $$ \alpha_i^{\text{new}} = \alpha_i^{\text{old}} + y_i y_j (\alpha_j^{\text{old}} - \alpha_j^{\text{new}}). $$

3. **Update Bias $\theta_0$**:  
   Recompute $\theta_0$ using the new α’s to satisfy KKT conditions.

4. **Convergence Check**:  
   Repeat until all α’s satisfy KKT within tolerance $\epsilon$ or max iterations reached.

##### Advantages
- **Efficiency**: Solves subproblems analytically, avoiding QP overhead.
- **Low Memory**: Only two columns of the kernel matrix needed per iteration.
- **Scalability**: Suited for large datasets.
- **Guaranteed Convergence**: By optimizing violating pairs.

##### Practical Considerations
- **Kernel Cache**: Stores frequently used kernel values to speed up computation.
- **Heuristics**:  
  - **First Choice**: Use a working set of non-bound (0 < αᵢ < C) examples.  
  - **Second Choice**: Pick αⱼ with the largest prediction error difference $|E_i - E_j|$.
- **Complexity**: Typically $O(m^2)$ to $O(m^3)$ per iteration, but efficient in practice.

##### Applications
SMO is widely used for:
- Binary classification (e.g., text categorization, image recognition).
- Extensions to regression ($\nu$-SVR) and multiclass SVMs.

##### Pseudo-code
![[Pasted image 20250724134544.png]]

### Kernel Trick

As comprehensive kernel trick is introduced ([[Kernel Trick]]), the primal form of SVM is turned into$$\begin{aligned}
\min \quad & \frac{1}{2}\|\mathbf{\theta}\|^2 + C\sum^n_{i=1}\xi_i\\
\text{s.t.} \quad &y_{i}\left(\varphi(\mathbf{x}_i)^T\mathbf{\theta}+\theta_0\right)\geq 1-\xi_i,\forall i=1,2,...,n\\
&\xi_i\geq0,\forall i=1,2,...,n
\end{aligned}$$With the dual form as$$\begin{aligned}
\min_\alpha\quad & \frac{1}{2}\sum^n_{t=1}\sum^n_{s=1}y_ty_s\alpha_t\alpha_sK\Bigl(\mathbf{x}_t,\mathbf{x}_s\Bigr)-\sum^n_{t=1}\alpha_i\\
\text{s.t.}\quad & \sum^n_{t=1}y_t\alpha_t=0\\
&0\leq\alpha_i\leq C,\forall i=1,2,...,n
\end{aligned}$$Where:
- $K(\mathbf{x},\mathbf{y})=\varphi(\mathbf{x})^T\varphi(\mathbf{y})$ is the kernel function of mapping $\varphi$.

## Codes

```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0)
model.fit(X_train, y_train)
```
*In scikit-learn, the `SVC` class uses an SMO-like algorithm (LibSVM).*
## Complement

1. Proof for the necessity of support vectors that influence the margin:
		According to the definition of support vector, $\alpha_i=0$ for non support vector. Then, we have $\gamma_i=C-\alpha_i=C\neq0$ and $\xi_i=\frac{0}{\gamma_i}=0$. Since $y_i(\left\langle\mathbf{\theta},\mathbf{x}\right\rangle+\theta_0)\geq1$, the constraint is satisfied. Therefore we have:
		- non-support vectors doesn't affect $\mathbf{\theta},\theta_0$
		- the constraints are already satisfied
		Therefore the non-support vectors do not influence the final result. ^eedf00