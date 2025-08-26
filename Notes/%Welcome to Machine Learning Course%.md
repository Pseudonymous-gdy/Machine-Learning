
**Document Class** #introduction #DSAA2011
# Overview:
**1. Supervised Learning**
[Supervised Learning: Regression and Classification](Supervised%20Learning.md)


**2. Model-level Learning, Evaluation, and Datasets**
[Feature Engineering](Feature%20Engineering.md)
[Model Evaluation and Choice](Model%20Evaluation%20and%20Choice.md)
[[Ensemble Methods]]
[[Feature Selection]]


**3. Unsupervised Learning**
[Clustering](Clustering.md)
[Dimension Reduction](Dimension%20Reduction.md)
[Markov and Graphical Models](Markov%20and%20Graphical%20Models.md)


**4. Reinforcement Learning**
[Active Learning & RL](Active%20Learning%20&%20RL.md)


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

View [[Mathematical Notes]] if necessary. Here we will primarily use Maximum Likelihood Estimator.