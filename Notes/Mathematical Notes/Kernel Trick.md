# Brief Introduction
Many real-world datasets are not linearly separable in their original space, making Euclidean Distance unlikely to perform well. Therefore, we would consider a mapping from lower dimensions to higher dimensions to make the datasets more linearly separable.

How to map vectors to higher dimensions? We consider $\varphi(\cdot)$ as a mapping from original vectors to vectors in different space. Then in the transformed space we have $\mathbf{\theta}^T\varphi(\mathbf{x})+\theta_0=0$. Things are changed once we adopt $\varphi(\mathbf{x})$ instead of $\mathbf{x}$. But how to describe such mapping functions? The answer is you don't have to.

The mapping function $\varphi$ is settled to be probably in infinite dimensions without shown representations, but to get a kernel function $K(x_1,x_2):=\varphi(x_1)^T\varphi(x_2)$ which needs to satisfy (Mercer):
- $K(x_1,x_2)=K(x_2,x_1)$
-  $\int K(x,y)g(x)g(y)dxdy\geq0,\forall g$ (positive semidefinite)
Some of the common kernel functions are:
-  $K(\mathbf{x}_1,\mathbf{x}_2)=e^{-\frac{||\mathbf{x}_1-\mathbf{x}_2||^2}{2\sigma^2}}$ (Gaussian Kernel)
-  $K(\mathbf{x}_1,\mathbf{x}_2)=(\mathbf{x}_1^T\mathbf{x}_2+c)^d$ (Polynomial Kernel)
-  $K(\mathbf{x}_1,\mathbf{x}_2)=\tanh(k\mathbf{x}_1^T\mathbf{x}_2-\delta)$ (Sigmoid Kernel)
-  $K(\mathbf{x}_1,\mathbf{x}_2)=\exp(-\gamma\|\mathbf{x}_1-\mathbf{x}_2\|)$ (Radial Basis Function Kernel)

# Analysis on Kernels

## RBF Kernel

Consider data points with:
- inner points labeled 0
- outer points labeled 1
Since it is obviously not linearly separable, RBF kernel is applied with several features:
- Maps to an infinite space
- Measures similarity based on distance
- $\gamma$ controls the ”reach” of the kernel.

# An example of XOR

Given data of XOR points in 2D: $$\begin{aligned}(0,0),(1,1)= Class 0\\(0,1),(1,0)= Class 1\end{aligned}$$We apply Polynomial Kernel of $K(x, y)=\left(x^{\top} y\right)^{2}$. This, in essence, helps a mapping to 3D that$$\begin{aligned}\phi\left(x_{1}, x_{2}\right)\quad\quad&=\quad\left(x_{1}^{2}, \sqrt{2} x_{1} x_{2}, x_{2}^{2}\right),\text{and}\\\Rightarrow (0,1) \mapsto(0,0,1) &;\quad(1,0) \mapsto(1,0,0)\\\Rightarrow (1,1) \mapsto(1, \sqrt{2}, 1) &;\quad(0,0) \mapsto(0,0,0)\end{aligned}$$Thus making the data points linearly separable.

# Clustering Applications

## Prototype-based Method
Consider [[K-Means and K-Medoids]]. In the traditional algorithm, clustering (e.g., Kernel K-Means) assumes linear separability in the original space. To improve performance on non-linearly separable data, we might think of using kernel tricks as distance measurement. Based on the raw idea, procedures are given by:
1. Uses a kernel function to compute distances or similarities in the transformed space.
2. Perform clustering (e.g., K-Means) in that space.

However, it is hard to determine the mapping function, and Kernel helps leaving out related calculations. Consider the loss function$$\begin{aligned}\mathcal{L}(\Delta)&=\sum^K_{k=1}\sum_{i\in C_k}\|\varphi(x_i)-\varphi(\mu_k)\|^2\\
&=\sum^K_{k=1}\sum_{i\in C_k}\left(\varphi(x_i)-\varphi(\mu_k)\right)^T\left(\varphi(x_i)-\varphi(\mu_k)\right)\\
&=\sum^K_{k=1}\sum_{i\in C_k}\varphi(x_i)^T\varphi(x_i)-\varphi(x_i)^T\varphi(\mu_k)-\varphi(\mu_k)^T\varphi(x_i)+\varphi(\mu_k)^T\varphi(\mu_k)\\
&=\sum^K_{k=1}\sum_{i\in C_k}K(x_i,x_i)-2K(x_i,\mu_k)+K(\mu_k,\mu_k)\\
\end{aligned}$$Which leads to Kernel K-Means [[K-Means and K-Medoids#Kernel Tricks]].