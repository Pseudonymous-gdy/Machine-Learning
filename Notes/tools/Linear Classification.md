
**Document Class** #DSAA2011  #Supervised-Learning #Classification 

**Table of Contents:**
- [[#Binary Classification|Binary Classification]]
- [[#Multi-class Classification|Multi-class Classification]]
- [[#Codes|Codes]]

**Linear Classification is derived from [[Linear Regression]].**

---
## Binary Classification

Here, we manually label the $y$ variable as $\{-1,1\}$, and conduct the linear regression.

<font color=red>Learning & Training</font>: Given a dataset $\{(\mathbf{x}_i,y_i)\}^m_{i=1}$ (where $y_i=\pm1$), learn the weights using least squares
$$
\mathbf{w}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}\in\mathbb{R}^{d+1}
$$
<font color=red>Prediction & Testing</font>: Given a new data sample $\mathbf{x}_{new}\in\mathbb{R}^d$, the predicted label is
$$
\hat{y}_{new}=\text{sign}(\mathbf{x}_{new}^T\mathbf{w})\in\{+1,-1\}
$$
**Note:** if the raw prediction is exactly 0, we declare an error.

## Multi-class Classification

Here, we conduct `one-hot encoding` since a direct assignment of values may cause:
- a further explanation in distance between assigned values
- a harder attempt to deal with intermediate values

To be detailed, the label value is a vector, like
$$
\mathbf{y}_i=\begin{bmatrix}
\mathbb{1}(sample\ i\ in\ class\ 1) & \mathbb{1}(sample\ i\ in\ class\ 2) & ... & \mathbb{1}(sample\ i\ in\ class\ C)
\end{bmatrix}
$$
If stacked, we have
$$
\mathbf{Y}=\begin{bmatrix}
\mathbf{y}_1 \\ \mathbf{y}_2 \\ ... \\ \mathbf{y}_m
\end{bmatrix}=\begin{bmatrix}
y_{1,1} & y_{1,2} & ... & y_{1,C}\\
y_{2,1} & y_{2,2} & ... & y_{2,C}\\
...&&&...\\
y_{m,1} & y_{m,2} & ... & y_{m,C}\\
\end{bmatrix}
$$
Similarly, we define how the model is used.

<font color=red>Training & Learning</font>:
If $\mathbf{X}$ has full column rank, then
$$
\mathbf{W}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}\in\mathbb{R}^{(d+1)\times C}
$$

<font color=red>Testing & Prediction</font>:
$$
\hat{y}^{new,reg}[k]=\Bigl[1\ \mathbf{x}_{new}^T\Bigr]\mathbf{W}[k]
$$
$$
\hat{y}_{new}=\arg\max_{k\in\{1,2,...,C\}}\hat{y}^{new,reg}[k]
$$
## Codes

The codes are similar with the linear regression. Namely, for binary classification:

```python
# with the same code as linear regression
y_class_predict = np.sign(y_predict)
```

For multi-class classification:

```python
# ...similar codes
from sklearn.preprocessing import OneHotEncoder
y_class = # an array of class
onehot_encoder = OneHotEncoder(sparse=False) # dtype of output matrix: sparse or sparse_output
Ytr_onehot = onehot_encoder.fit_transform(y_class)
# ...train and prediction...
X_test = np.array(...)
yt_est = X_test @ W
class_ytest = np.argmax(yt_est)+1
```