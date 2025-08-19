# Type of Data

`Type of Data:`
- Support Distinctness: Nominal
- Support Order: Ordinal
- Support Addition: Interval
- Support Multiplication: Ratio
*Note: Support Multiplication means that the variable is meaningful if multiplies or divided by any real number.*

`General Characteristics of Dataset:`
- **Dimensionality**: Number of attributes that the objects in the dataset possess. *(Curse of dimensionality: difficulties associated with the analysis of high-dimensional data.)*
- **Distribution**: Frequency of occurenc3e of various values or set of values for the attributes comprising data objects. *(Skewness, Sparsity-few values)*
- **Resolution (分辨率)**: The resolution of an image is how clear the image is. Fine resolution$\rightarrow$not visible pattern; Coarse resolution$\rightarrow$disappeared pattern.

`Taxonomy:`
- **Record Data**: a collection of records, no explicit relationship among records or data fields, every record has same set of attributes.
	- **Transaction or Market Basket Data**: each transaction involves a set of items (you may buy multiple items in a grocery store).
	- **Data Matrix**: data objects in a collection of data have the same fixed set of numeric attributes$\rightarrow$vectors in a multidimensional space$\rightarrow$could be put into a matrix.
	- **Sparse Data Matrix**: attributes are of same type and are asymmetric, i.e., only non-zero values are important.
	- **Graph Based Data**: data objects as nodes in the graph *(Clear Relationship)*; data with objects that are graph *(sub objects relationships)*.
	- **Ordered Data**: relationships involving order in time or space *(e.g., sequential transaction data, time series data, sequence data, spatial and spatio-temporal data)*

# Data Quality
## Measurement and Collection Issues

`Measurement Error`: the value recorded differs from the true value to some extent.
`Data Collection Error`: errors caused by data collection e.g. omitting data objects, or attribute values, or inappropriately including a data object.

**ISSUES (ERRORS)**:
- **Noise and Artifacts**: random component of a measurement error; deterministic distortions of the data.
- **Precision, Bias, and Accuracy**: closeness of repeated measurements; systematic variation of measurements from the quantity being measured; closeness of measurements to the true value of the quantity being measured.
- **Outliers**: characteristics/values of attributes abnormal.
- **Missing Values**: miss one or more attribute values. *Resolution: eliminate/estimate/ignore*
- **Inconsistent Values**: Data inconsistent logically. *i.e., zip code$\neq$city*
- **Duplicate Data**: include data objects that are (almost) duplicates. *Deduplication.*

**ISSUES RELATED TO APPLICATIONS**：
- Timeliness: whether the data is still valid at present
- Relevance: whether the data contains necessary information *(i.e., sampling bias)*
- Knowledge about the data: information contained in the data *(relations between attributes, lack in documentation, etc.)*

# Data Preprocessing

1. Aggregation
	- `Definition`: combining of two or more objects.
	- `Motivation`:
		- less memory and processing time
		- change of scope or scale by elaborating a high-level view of the data
2. Sampling: Reduce computational expenses
	- sampling with/without replacement, stratified, etc.
	- adaptive/progressive sampling: start from small samples, and progress to get the correct one.
3. Dimensionality Reduction