
**Document Class** #DSAA2011  #Mathematics 

**Table of Contents:**
- [[#Motivation|Motivation]]
- [[#Methodology|Methodology]]
- [[#Choice of Convergence Criteria|Choice of Convergence Criteria]]
- [[#Learning Rate|Learning Rate]]
	- [[#Learning Rate#Thoughts 1: Decreasing Learning Rate.|Thoughts 1: Decreasing Learning Rate.]]
	- [[#Learning Rate#Thoughts 2: Adagrad (Adaptive Gradient Algorithm).|Thoughts 2: Adagrad (Adaptive Gradient Algorithm).]]
- [[#Variations|Variations]]
	- [[#Variations#Different Gradients|Different Gradients]]
		- [[#Different Gradients#1. Momentum-based GD|1. Momentum-based GD]]
		- [[#Different Gradients#2. Nesterov Accelerated Gradient (NAG)|2. Nesterov Accelerated Gradient (NAG)]]
	- [[#Variations#View of dataset|View of dataset]]
		- [[#View of dataset#1. Batch Gradient Descent (BGD)|1. Batch Gradient Descent (BGD)]]
		- [[#View of dataset#2. Stochastic Gradient Descent (SGD)|2. Stochastic Gradient Descent (SGD)]]
		- [[#View of dataset#3. Mini-Batch Gradient Descent (MBGD)|3. Mini-Batch Gradient Descent (MBGD)]]

**The simplest way for solving optimization tasks.**

---
## Motivation

What if having an irregular loss function to minimize/maximize?
- Iteratively estimate $\mathbf{w}$.
- Optimization workhorse for modern machine learning is gradient descent

## Methodology

Suppose we want to minimize $C(\mathbf{w})$ with respect to $\mathbf{w}=[w_1,...,w_d]^T$. Then, for the gradient, we have
$$\nabla_\mathbf{w}C(\mathbf{w})=\begin{pmatrix}\frac{\partial C}{\partial w_1}\\\frac{\partial C}{\partial w_2}\\...\\\frac{\partial C}{\partial w_d}\end{pmatrix}$$
Then, we state that (proof omitted):
- $\nabla_{\mathbf{w}}C(\mathbf{w})$ is the direction at $\mathbf{w}$ where $C$ is increasing most rapidly
- $-\nabla_{\mathbf{w}}C(\mathbf{w})$ is the direction at $\mathbf{w}$ where $C$ is decreasing most rapidly

Based on the previous thought, we may get the iteration formula:
$$
\mathbf{w}_{k+1}\gets \mathbf{w}_k-\eta\nabla_{\mathbf{w}}C(\mathbf{w}_k)
$$

Where $\eta$ is the learning rate, and we conduct the algorithm as follows.

`Algorithm: Gradient Descent`:
- Initialize $\mathbf{w}_0$ and learning rate/step size $\eta>0$
- **while** <font color=red>true</font> **do**
	- Compute $\mathbf{w}_{k+1}\gets\mathbf{w}_k-\eta\nabla_\mathbf{w}C(\mathbf{w}_k)$
	- **If** converge **then**
		- **return** $\mathbf{w}_{k+1}$
	- **end if**
- **end while**

There are some thing to notice:
- $\eta$ is a hyperparameter that should be chosen ahead of the training.
- the method goes smoothly if $\eta$ is not too big, leading to $C(\mathbf{w}_{k+1})<C(\mathbf{w}_k)$, and a better estimate of $\mathbf{w}$ is got after each iteration.
- The above is about to get the **minimum**. If to get the maximum, the sign of the gradient should be positive.

`Terminal State`: there are some possible convergence criteria to stop iteration.
- Set maximum iteration $k_{max}$.
- Check `percentage` or `absolute change` in objective function $C$ below a threshold
- Check `percentage` or `absolute change` in parameter $\mathbf{w}$ below a threshold

**Note: Gradient descent can only find local minimum.** This is because gradient$=\mathbf{0}$ at local minimum, where $\mathbf{w}$ will concentrate instead of change when approaching the local minimum.

## Choice of Convergence Criteria

Review about the terminal state, which mentions about the convergence criteria, and we focus on the pros and cons of each (instead of the first one).

**Check `percentage` or `absolute change` in objective function $C$ below a threshold**:
- Pro:
	- direct measure of optimization progress
	- prevents unnecessary iterations
- Con:
	- computational cost (calculate the loss function)
	- sensitive to scale (if for absolute change, the loss function varies a lot while $\mathbf{w}$ does not)
	- NOT guarantee the convergence of $\mathbf{w}$


**Check `percentage` or `absolute change` in parameter $\mathbf{w}$ below a threshold**:
- Pro:
	- Stable solution
	- Less sensitive to oscillations (振荡) in the function value
- Con:
	- Longer time to detect convergence ($\mathbf{w}$ varies a lot while the loss function does not)
	- NOT directly reflect progress in objective function

`Conclusion` - Changes of function $C$ or parameter $\mathbf{w}$:
- Function $C$:
	- care primarily about minimizing $C$
	- $C$ is well-behaved (e.g., smooth and convex)
	- low computational cost
- Parameter $\mathbf{w}$:
	- care primarily if the parameters are stable and the optimization process has converged

`Do Hybrid`: set different conditions for convergence, e.g.
$$
\|\mathbf{w}_{k+1}-\mathbf{w}_k\|<\delta,\|C(\mathbf{w}_k)-C(\mathbf{w}_k)\|<\varepsilon
$$
It might take longer time, but may get a better converged result.

## Learning Rate

`Considerations`:
- **When $\eta$ is too small**: converge slowly
- **When $\eta$ is too large**: overshoot the local, probably leading to slow convergence or non-convergence.

Therefore, we consider `changing learning rate`. Thoughts are introduced in the following.

### Thoughts 1: Decreasing Learning Rate.
This is motivated when in the early stage, a speeded-up convergence is need, while in the later stage, the solution should be rather refined. The formula is given below. **However, remember to have sanity check** (labeled red).

$$\begin{aligned}
\mathbf{w}_{k+1}&\gets \mathbf{w}_k-\eta_k\nabla_{\mathbf{w}}C(\mathbf{w}_k)\\
\eta_{k+1}&\gets\frac{\eta_k}{\alpha}\text{ or }\eta_k-\alpha\text{ or ... }\textcolor{red}{>0}
\end{aligned}$$

### Thoughts 2: Adagrad (Adaptive Gradient Algorithm).
The formula is given below.

$$\begin{aligned}
\mathbf{w}_{k+1}&\gets \mathbf{w}_k-\eta\left(\mathbf{M}_k+\varepsilon\mathbf{I}\right)^{-1}\nabla_{\mathbf{w}}C(\mathbf{w}_k)\\
\mathbf{M}_k&=\operatorname{diag}\left(\sum^k_{s=0}\nabla_{\mathbf{w}}C(\mathbf{w}_s)\odot\nabla_{\mathbf{w}}C(\mathbf{w}_s)\right)
\end{aligned}$$
Where $\mathbf{M}_k$ is a diagonal matrix where each diagonal entry is the sum of the element-wise squared gradients, and $\varepsilon$ is a small constant to prevent division by zero, $\operatorname{diag}$ is the diagonalization of a vector into a diagonal matrix. For element-wise view, we could see

$$
\left[\mathbf{w}_{k+1}\right]_i\gets\left[\mathbf{w}_k\right]_i-\frac{\eta}{\mathbf{M}_{ii,k}+\varepsilon}\left[\nabla_{\mathbf{w}}C(\mathbf{w}_k)\right]_i\text{, where }\mathbf{M}_{ii,k}=\sum^k_{s=0}\left[\nabla_{\mathbf{w}}C(\mathbf{w}_s)\right]^2_i
$$

`Pros & Cons`:
- Pros: this adjusts the learning rate for based on the historical gradient information, thus giving larger update to $\mathbf{w}_k$ with smaller gradient and smaller update to $\mathbf{w}_k$ with larger gradient.
- Cons: learning rate tends to shrink too quickly, leading to slow convergence or getting stuck at suboptimal points (vanishing learning rate).

The above all leads us to reflect on better strategy of gradient descent, in the next section.

## Variations

Given the standard gradient:
$$
\mathbf{w}_{k+1}\gets \mathbf{w}_k-\eta\nabla_{\mathbf{w}}C(\mathbf{w}_k)
$$
We explore variations of gradient formula.

### Different Gradients
#### 1. Momentum-based GD

$$
\mathbf{v}_k\gets\beta \mathbf{v}_{k-1}+(1-\beta)\nabla_\mathbf{w}C(\mathbf{w}_k),\mathbf{w}_{k+1}\gets\mathbf{w}_k-\eta \mathbf{v}_k
$$
Where $\beta$ is the momentum factor, $v_0=\nabla_{\mathbf{w}}C(\mathbf{w}_0)$.

`Pros & Cons`:
- Pros: maintaining a **moving average** of past gradients; fast convergence.
- Cons: may overshoot the minimum.

#### 2. Nesterov Accelerated Gradient (NAG)

$$
\mathbf{v}_k\gets\beta\mathbf{v}_{k-1}+\eta\nabla_\mathbf{w}C(\mathbf{w}_k-\beta\mathbf{v}_{k-1}), \mathbf{w}_{k+1}\gets\mathbf{w}_k-\mathbf{v}_k
$$

Anticipate the direction the optimizer will go in the next step. (Apply the term $\eta\nabla_\mathbf{w}C(\mathbf{w}_k-\beta\mathbf{v}_{t-1})$ to predict the further situation)

`Pros & Cons`:
- Pros: faster convergence
- Cons: slightly more complex

### View of dataset

#### 1. Batch Gradient Descent (BGD)

The loss function is built using the **entire dataset** at each iteration.

`Algorithm: Batch Gradient Descent (BGD)`:
- Initialize $\mathbf{w}_0$ and learning rate/step size $\eta>0$
- **while** <font color=red>true</font> **do**
	- Compute $$\mathbf{w}_{k+1}\gets\mathbf{w}_k-\eta\nabla_\mathbf{w}C(\mathbf{w}_k)$$
	- **if** converge **then**
		- **return** $\mathbf{w}_{k+1}$
	- **end if**
- **end while**

`Pros & Cons`:
- Pros: converges smoothly to the minimum
- Cons: computationally expensive, memory intensive, not ideal for online learning (hard to update the model incrementally as new data comes, which might only be dealt with from the very beginning)


#### 2. Stochastic Gradient Descent (SGD)

The loss function built using one **randomly** (probably indeterministic) chosen sample at each iteration

`Algorithm: Stochastic GD (SGD)`:
- Initialize $\mathbf{w}_0$ and learning rate/step size $\eta>0$
- **while** <font color=red>true</font> **do**
	- <font color=blue>Randomly choose one sample</font> $\textcolor{blue}{m}$
	- Build the loss function $C_m(\mathbf{w})$
	- Compute $$\mathbf{w}_{k+1}\gets\mathbf{w}_k-\eta\nabla_\mathbf{w}C_m(\mathbf{w}_k)$$
	- **if** converge **then**
		- **return** $\mathbf{w}_{k+1}$
	- **end if**
- **end while**

`Pros & Cons`:
- Pros: fast updates, suitable for online learning
- Cons: noisy update, longer convergence time

Note: here we might require a check for convergence of $\mathbf{w}$ as the convergence criteria since loss function is built within each iteration.

#### 3. Mini-Batch Gradient Descent (MBGD)

Divide the entire dataset into small mini-batch, and function is chosen on the mini-batches.

`Algorithm: Mini-Batch GD (MBGD)`:
- Initialize $\mathbf{w}_0$ and learning rate/step size $\eta>0$
- <font color=blue>Divide the entire dataset into small mini-batches</font>
- **while** <font color=red>true</font> **do**
	- <font color=blue>Randomly choose one small mini-batch</font> $\textcolor{blue}{n}$
	- Build the loss function $C_n(\mathbf{w})$ with the <font color=blue>chosen mini-batch</font> $\textcolor{blue}{n}$
	- Compute $\mathbf{w}_{k+1}\gets\mathbf{w}_k-\eta\nabla_\mathbf{w}C(\mathbf{w}_k)$
	- **if** converge **then**
		- **return** $\mathbf{w}_{k+1}$
	- **end if**
- **end while**

`Pros & Cons`:
- Pros: between SGD and BGD
- Cons: still has noise, hyperparameter tuning (choice of batch size)
