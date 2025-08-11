
`Decision Tree` is a simple, tree-like model that make decisions by asking a series of questions (usually yes/no). It could be applied for classification or regression. A plain example is as below.

![[Pasted image 20250729200948.png]]

# Basic Idea of Decision Tree

For example, imaging we're going to build a classifier on a dataset to predict if a flight will be normal, delayed, or cancelled ([DSAA1001_Project_DecisionTree_Demo - Colab](https://colab.research.google.com/drive/1bZcryXzMgBun7aheC2LptQvX45cTy1od#scrollTo=nMILeAyctVX7)). And here are a series of characteristics:
- Day of Month, Day of Week, Flight Date, Unique Carrier, Airline ID,...
- Origin Airport, City, State,...
- Destination Airport, City,...
- Departure Time, Arrival Time,...
- Distance,...

How do we determine about the flight status? You might ask for the departure time and arrival time so as to compare with the normal or delayed flight to get the result, and in the case probably a time over a certain threshold might be detected as abnormal. That is what decision tree does-it `extracts features from observations` and find out about the features that could help predict the class a flight is in. Basically, the tree has **`structures`** of:

1. **Root Node**: _This_ _is_ _the_ _starting_ _point_ _of_ _the_ _tree_ _and_ _represents_ _the_ _entire_ _dataset_. _It_ _evaluates_ _all_ _features_ _to_ _determine_ _the_ _best_ _initial_ _split_, _often_ _based_ _on_ _metrics_ _like_ _information_ _gain_ _or_ _Gini_ _impurity_.
2. **Decision Nodes**: _These_ _are_ _internal_ _nodes_ _where_ _the_ _dataset_ _is_ _further_ _split_ _based_ _on_ _specific_ _conditions_. _Each_ _decision_ _node_ _evaluates_ _a_ _feature_ _and_ _branches_ _the_ _data_ _into_ _subsets_, _aiming_ _to_ _increase_ _homogeneity_ _within_ _each_ _subset_.
3. **Leaf Nodes**: _These_ _are_ _terminal_ _nodes_ _that_ _represent_ _the_ _final_ _output_ _or_ _decision_. _In_ _classification_ _tasks_, _they_ _correspond_ _to_ _class_ _labels_, _while_ _in_ _regression_ _tasks_, _they_ _provide_ _predicted_ _values_.

![[Pasted image 20250729210901.png]]

By differentiating through features, we hope decision-tree could return a final answer. Specially, for classification tasks, we hope the leaf node to be as pure as possible. That is, we hope there could be only one kind of answers when decision tree asks all about the conditions. This comes to how we adapt such thoughts.

# How to Build a Decision Tree

## Sketch

**A decision tree builds by**:
1. Start with a dataset $\mathcal{D}$ (with features like age, income, etc.)
2. Pick a feature to split on (e.g., "Is age > 30?")
3. Keep splitting based on features until a decision is reached

This comes with a **`GOAL`** of:
**Minimizing Uncertainty at each step (using metrics like entropy or Gini impurity)**. Basically, we have
$$\begin{aligned}
&\square\ p(j|t):\quad\ \text{relevant frequency of class j at node t}\\
&\blacktriangle Entropy: \operatorname{Entropy}(t)=-\sum_jp(j|t)\log p(j|t)\\
&\blacktriangle Gini\ impurity: \operatorname{GINI}(t)=1-\sum_j[p(j|t)]^2
\end{aligned}$$

## Detail

***Given a subset of data $M$ (a node in a tree), decision trees conduct a $\textcolor{red}{\text{Greedy Algorithm}}$ by:***

- **For each feature $h_i(x)$**:
	- **Split** data of $M$ according to feature $h_i(x)$;
	- **Compute classification error** of split.
	
- **Choose feature $h^*(x)$ with $\textcolor{red}{\text{lowest}}$ classification error** (or other criteria, e.g., `Gini Gain`, `Entropy Gain`, `Mean Squared Error`, etc.)
	- `Gini Gain`: $\operatorname{Gain}_{\text{GINI}}(M,h_i(x))=\operatorname{GINI}(M)-\frac{|M_1|}{M}\operatorname{GINI}(M_1)-\frac{|M_2|}{M}\operatorname{GINI}(M_2)-...$
	- `Entropy Gain`: $\operatorname{Gain}_{\text{Entropy}}(M,h_i(x))=\operatorname{Entropy}(M)-\frac{|M_1|}{M}\operatorname{Entropy}(M_1)-\frac{|M_2|}{M}\operatorname{Entropy}(M_2)-...$
	
- **Stop when `stopping conditions` are met**:
	- `Purity`: All data in a node belongs to one class.
	- `Max Depth`: Limit how many levels the tree can have.
	- `Min Samples`: Stop if too few data points remain.
	- (These are aimed at *PREVENTING OVERFITTING AND KEEPING THE TREE MANAGABLE*.)
	

## Example:

We are going to execute a decision tree on the dataset:

![[Pasted image 20250729211319.png]]

For the first layer, we calculate the gain of each features:
$$\begin{aligned}
\operatorname{Entropy}(Play\ Golf)&=\operatorname{Entropy}(0.36,0.64)=0.94\\
\operatorname{Gain}(Outlook)&=\operatorname{Entropy}(Play\ Golf)-P(Sunny)\operatorname{Entropy}(3,2)-P(Overcast)\operatorname{Entropy}(4,0)\\&\quad-P(Rainy)\operatorname{Entropy}(2,3)=0.247\\
\operatorname{Gain}(Temp)&=0.029\\
\operatorname{Gain}(Humidity)&=0.152\\
\operatorname{Gain}(Windy)&=0.048
\end{aligned}$$
The calculation is based on the count table of:

|              |          | Play | Not Play |           |       | Play | Not Play |
| :----------: | :------: | :--: | :------: | :-------: | :---: | :--: | :------: |
| **Outlook**  |  Sunny   |  3   |    2     | **Temp**  |  Hot  |  2   |    2     |
|              | Overcast |  4   |    0     |           | Mild  |  4   |    2     |
|              |  Rainy   |  2   |    3     |           | Cool  |  3   |    1     |
| **Humidity** |   High   |  3   |    4     | **Windy** | False |  6   |    2     |
|              |  Normal  |  6   |    1     |           | True  |  3   |    3     |

Similarly, for the rest layers, it could also be conducted.

# Pros and Cons

`Pros`:
- Can handle large datasets
- Can handle mixed predictors (continuous, discrete, qualitative)
- Can ignore redundant variables
- Can easily handle missing data
- Easy to interpret if small

`Cons`:
- Prediction performance is poor
- Does not generalize well
- Large trees are hard to interpret


# Codes

## Classification:

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
```

## Regression:

```python
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=42)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
```

## Demo:

```python
# Classifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
cross_val_score(clf, iris.data, iris.target, cv=10)
# output:array([ 1.     ,  0.93,  0.86,  0.93,  0.93,
#        0.93,  0.93,  1.     ,  0.93,  1.      ])

# Regressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
X, y = load_diabetes(return_X_y=True)
regressor = DecisionTreeRegressor(random_state=0)
cross_val_score(regressor, X, y, cv=10)
# output:
# array([-0.39, -0.46,  0.02,  0.06, -0.50,
#       0.16,  0.11, -0.73, -0.30, -0.00])
```

# Credits

Li, J. (2024, November). Decision Tree \[PowerPoint Slides\]. Thrust of Data Science and Analytics, The Hong Kong University of Science and Technology (Guangzhou).

Koli, S. (2023, February). *Decision Trees: A Complete Introduction With Examples*. Medium. https://medium.com/@MrBam44/decision-trees-91f61a42c724.