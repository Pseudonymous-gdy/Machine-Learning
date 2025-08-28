
**Document Class** #DSAA2011  #Preprocessing #introduction 

**Table of Contents:**
- [[#Recall: Polynomial Transformation|Recall: Polynomial Transformation]]
- [[#Notation: Boolean, Nominal, Ordinal Features|Notation: Boolean, Nominal, Ordinal Features]]
	- [[#Notation: Boolean, Nominal, Ordinal Features#Boolean Variables|Boolean Variables]]
	- [[#Notation: Boolean, Nominal, Ordinal Features#Nominal Values: One-hot Encoding|Nominal Values: One-hot Encoding]]
	- [[#Notation: Boolean, Nominal, Ordinal Features#Ordinal Values:|Ordinal Values:]]
- [[#Missing Values|Missing Values]]
- [[#Non-linear Transformation|Non-linear Transformation]]
- [[#Be Creative! Further techniques|Be Creative! Further techniques]]

**A Beginning of Model Uninterpretability.**

---
## Recall: Polynomial Transformation

Refer to [[Polynomial Regression]] if necessary.

As a simple example, this indicates a further discussion on feature engineering.

`Feature Engineering`: the process of using domain knowledge to extract features (characteristics, properties, attributes) from raw data.

In the polynomial regression example, we are actually conducting linear regression with mapped vectors using manually-added monomial features.

## Notation: Boolean, Nominal, Ordinal Features

### Boolean Variables

We define Boolean indicator function first.

`Boolean Indicator Function`: 
$$\mathbb{1}(\mathcal{A}):=\begin{cases}1, \mathcal{A}\text{ is true}\\0, \mathcal{A}\text{ is false}\end{cases}$$

It's a mapping of $\phi:\mathcal{X}=\{\textcolor{blue}{true},\textcolor{red}{false}\}\to\{0,1\}:\phi(x)=\mathbb{1}(x)$.

Then for $\mathcal{X}=\{\textcolor{blue}{true},\textcolor{red}{false}\}^2$, define: $\phi(\mathbf{x}):=[\mathbb{1}(x_1),\mathbb{1}(x_2),\mathbb{1}(x_1\text{ and }x_2),\mathbb{1}(x_1\text{ or }x_2)]$, with similarities with polynomials in $[\mathbb{1}(\mathbf{x}_1),\mathbb{2}(\mathbf{x}_2)]$ span the same space. This encodes logical expressions.


### Nominal Values: One-hot Encoding

Refer to [[Linear Classification#Multi-class Classification]] if necessary.

This assign Boolean indicators to each category in this variable.

i.e., $\phi(x)=[\mathbb{1}(x=apple),\mathbb{1}(x=banana),\mathbb{1}(x=orange)]$.

### Ordinal Values:

There are two ways to conduct the coding:

`Real Encoding`:
$$
\phi(x)=\begin{cases}1,x=Class\ I\\2,x=Class\ II\\...\end{cases}
$$

`Boolean Encoding`:

$$
\phi(x)=[\mathbb{1}(x\geq Class\ I),\mathbb{1}(x\geq Class\ II),...]
$$

## Missing Values

There are different strategies when handling this:

- Remove Rows/Columns with Missing Entries
- (For time series) back-fill with most recent observed value
- Impute with mean, median, or mode
- Fancier imputation method: matrix completion, deep learning, ...
- Add new feature: Boolean indicator $\mathbb{1}(data\ is\ missing)$
	- can detect if missingness is informative
	- can complement imputation method
	- can use different indicators for different kinds of missingness


## Non-linear Transformation

Sometimes data is easy to predict with a simple but nonlinear relation, e.g.,
$$
\log(y)=\mathbf{w}^T\mathbf{x}\quad\text{equivalent to}\quad y=\exp(\mathbf{w}^T\mathbf{x})
$$

May transform $\mathbf{x}$ or $y$. By conducting nonlinear transforms like log, exp, quantile, median, etc., could help us handle better about the prediction tasks. Also, deep learning offers alternative way for non-linear transformation.

## Be Creative! Further techniques
