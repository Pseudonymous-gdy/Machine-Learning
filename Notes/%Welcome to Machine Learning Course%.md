# Overview:
**1. Supervised Learning**
[Supervised Learning: Regression and Classification](Supervised%20Learning)
[Feature Engineering](Feature%20Engineering)
[Model Evaluation and Choice](Model%20Evaluation%20and%20Choice)


**2. Unsupervised Learning**
[Clustering](Clustering)
[Dimension Reduction](Dimension%20Reduction)
[Markov and Graphical Models](Markov%20and%20Graphical%20Models)


**3. Reinforcement Learning**
[Active Learning & RL](Active%20Learning%20&%20RL)


# Taxonomy:

`Machine Learning`: A computer program is said to learn from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$.

*Note:*
- Experience: data, like games played by individual
- Performance measure: winning rate
- Task: win

`Taxonomy`:
Unsupervised: natural clusters or dimensions in the data. *(unlabeled data)*
- no outcome information available
- independently identify groups without labels or instruction
- characteristics that define groups

Supervised: system learn from given dataset to determine what actions to take in different situations. *(labeled data)*
- outcome information
- find patterns relate to outcomes
- use patterns to predict outcomes
	*Note:* regression: continuous; classification: discrete

RL: Active learn by the system's own actions, which affect situations in the future. *(env feedback)*
- makes decision based on trial and error
- decision-making algorithm is refined based on "rewards"
- excels in complex situations

# Mathematical Tools:

- MLE:
	- Efficiency: MLE is efficient, achieving the lowest possible variance among all unbiased estimators.
	- Asymptotic Normality: $\sqrt{n}(\hat{\theta}_{ML}-\theta)\rightarrow^d\mathcal{N}(0,I(\theta)^{-1})$
	- Consistency, sample size increases to infinity, the estimator will converge to the true parameter value
	- Unbiasedness: MLE is not necessarily unbiased, but in certain cases, it can be unbiased.
		Counterexample: $X_1,X_2,...,X_m\sim\exp{(\theta)}$, then:$$\begin{aligned}
		\mathbb{E}[\hat{\theta}]&=m\cdot\mathbb{E}\bigl[\frac{1}{\sum X_i}\bigr]\\
		&=m\cdot\int^{+\infty}_{0}\frac{1}{x}\cdot{gamma(x)}dx\ \text{(Sum of m i.i.d $\exp{\theta}$ follows Gamma($m,\theta$))}\\
		&=\frac{m\theta}{m-1}\neq\theta
		\end{aligned}$$
	- Model Sensitivity: model assumptions (affect likelihood functions)