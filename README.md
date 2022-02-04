# Intermediate-Machine-Learning-Kaggle
Handle missing values, non-numeric values, data leakage, and more.

Introduction in [exercise-introduction.ipynb](https://github.com/gabboraron/Intermediate-Machine-Learning-Kaggle/blob/main/exercise-introduction.ipynb)

This course also available on YouTube: [George Zoto - Kaggle Mini Courses - Intermediate Machine Learning ](https://www.youtube.com/watch?v=T88D8HtuV4A) or by [
1littlecoder - Kaggle 30 Days of ML - Day 12 - Kaggle Missing Values, Encoding - Learn Python ML in 30 Days](https://www.youtube.com/watch?v=8CeFfzr1fAQ)


## Missing Values
### 1) A Simple Option: Drop Columns with Missing Values
> Unless most values in the dropped columns are missing, the model loses access to a lot of (potentially useful!) information with this approach. As an extreme example, consider a dataset with 10,000 rows, where one important column is missing a single entry. This approach would drop the column entirely!

```Python
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
```
```
Out:
MAE from Approach 1 (Drop columns with missing values):
183550.22137772635
```

### 2) A Better Option: Imputation
> **Imputation fills in the missing values with some number.** For instance, we can fill in the mean value along each column.
>
>  The imputed value won't be exactly right in most cases, but it usually leads to more accurate models than you would get from dropping the column entirely.

1. To make it, use [`SimpleImputer()`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) from [scikit-learn](https://scikit-learn.org/stable/index.html). You can use parameters such as: 
    - `strategy=` which can be `mean`, `median`, `most_frequent` which calculates based on each column or `constant`
    - `missing_values` which set what type will be the replacement: `int`, `float`, `str`, `np.nan` or `None`, by default: `default=np.nan`
2. on the returned table you can use [`fit_transform(input_samples)`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer.fit_transform) to fit to data, then transform it, and returns a transformed version of `input_samples`
   - also you can use just [`transform(X)`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer.transform) which will impute all missing values in `X` and returns `X` with imputed values.

For better understanding:
```Python
import numpy as np
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean') # For pandas dataframes with nullable integer dtypes 
                                                                 # with missing values, missing_values should be set to
                                                                 # np.nan, since pd.NA will be converted to np.nan.
imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])  # Fit the imputer on X
                                                       # now imp_mean is a SimpleImputer()
                                                       
X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
print(imp_mean.transform(X))
```
```
Out:
[[ 7.   2.   3. ]
 [ 4.   3.5  6. ]
 [10.   3.5  9. ]]
```
*Here we got `3.5` because we used `strategy='mean'` which calcualtes [arithmetic mean](https://en.wikipedia.org/wiki/Arithmetic_mean) of each column, what was in this case the average of <img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}%202%20\\%20nan%20\\%205%20\end{bmatrix} " /> column.*

In [our housing example dataset](https://www.kaggle.com/c/home-data-for-ml-course) this will be: 
```Python
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
```
```
Out:
MAE from Approach 2 (Imputation):
178166.46269899711
```

### 3) An Extension To Imputation
> Imputation is the standard approach, and it usually works well. However, imputed values may be systematically above or below their actual values (which weren't collected in the dataset). Or rows with missing values may be unique in some other way. In that case, your model would make better predictions by considering which values were originally missing.
>
> In this approach, we impute the missing values, as before. And, additionally, for each column with missing entries in the original dataset, we add a new column that shows the location of the imputed entries.
>
> In some cases, this will meaningfully improve results. In other cases, it doesn't help at all.

```Python
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
```

```
Out:
MAE from Approach 3 (An Extension to Imputation):
178927.503183954
```

#### why did imputation perform better than dropping the columns?
> The training data has 10864 rows and 12 columns, where three columns contain missing data. For each column, less than half of the entries are missing. Thus, dropping the columns removes a lot of useful information, and so it makes sense that imputation would perform better.

more about this in [exercise-missing-values.ipynb](https://github.com/gabboraron/Intermediate-Machine-Learning-Kaggle/blob/main/exercise-missing-values.ipynb)

## Categorical Variables
- Consider a survey that asks how often you eat breakfast and provides four options: *"Never"*, *"Rarely"*, *"Most days"*, or *"Every day"*. In this case, the data is categorical, because responses fall into a fixed set of categories.
- If people responded to a survey about which what brand of car they owned, the responses would fall into categories like *"Honda"*, *"Toyota"*, and *"Ford"*. In this case, the data is also categorical.

Unfortunately the models can't understand this categories, so we need to transform them.

> #### Define Function to Measure Quality of Each Approach
> Using `score_dataset()` to compare the three different approaches to dealing with categorical variables. This function reports the mean absolute error (MAE) from a random forest model. ***Keep in mind: in general, we want the MAE to be as low as possible!***
> 
> [On the Melbourne dataset](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot/home) this look like:
>
> Based on data type (`dtype`) we obtain a list of all of the categorical variables in the training data. For this dataset, the columns with text indicate categorical variables, these are `Object`s type now.
> ```Python
> # Get list of categorical variables
> s = (X_train.dtypes == 'object')
> object_cols = list(s[s].index)
>
> print("Categorical variables:")
> print(object_cols)
> ```
> ```
> Out:
>
> Categorical variables:
> ['Type', 'Method', 'Regionname']
> ```

### 1) Drop Categorical Variables
The easiest approach to dealing with categorical variables is to simply remove them from the dataset. This approach will only work well if the columns did not contain useful information.

We drop the `object` dtype columns with the `select_dtypes()` method.
```Python
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
```
```
Out:
MAE from Approach 1 (Drop categorical variables):
175703.48185157913
```

### 2) Ordinal Encoding
Ordinal encoding assigns each unique value to a different integer
```
Every day  ->   3 
never      ->   0
rarely     ->   1
most days  ->   2
never      ->   0
```
> This assumption makes sense in this example, because there is an indisputable ranking to the categories. Not all categorical variables have a clear ordering in the values, but we refer to those that do as ordinal variables. For tree-based models (like decision trees and random forests), you can expect ordinal encoding to work well with ordinal variables.

[Scikit-learn](https://scikit-learn.org/stable/) has a [`OrdinalEncoder` class](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) that can be used to get ordinal encodings. We loop over the categorical variables and apply the ordinal encoder separately to each column.

For each column, we randomly assign each unique value to a different integer. This is a common approach that is simpler than providing custom labels; however, we can expect an additional boost in performance if we provide better-informed labels for all ordinal variables.

```Python
from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
```
```
Out:
MAE from Approach 2 (Ordinal Encoding):
165936.40548390493
```

### 3) One-Hot Encoding
One-hot encoding creates new columns indicating the presence (or absence) of each possible value in the original data. To understand this, we'll work through an example.
```
Color       =>   Red    Yellow    Green
Red               1        0        0 
Red               1        0        0
Yellow            0        1        0
Green             0        0        1
Yelow             0        1        0
```
*In the original dataset, "Color" is a categorical variable with three categories: "Red", "Yellow", and "Green". The corresponding one-hot encoding contains one column for each possible value, and one row for each row in the original dataset. Wherever the original value was "Red", we put a 1 in the "Red" column; if the original value was "Yellow", we put a 1 in the "Yellow" column, and so on.*

> **In contrast to ordinal encoding, one-hot encoding does not assume an ordering of the categories.** Thus, you can expect this approach to work particularly well if there is no clear ordering in the categorical data *(e.g., "Red" is neither more nor less than "Yellow")*. We refer to categorical variables without an intrinsic ranking as nominal variables.
>
> **One-hot encoding generally does not perform well if the categorical variable takes on a large number of values** *(i.e., you generally won't use it for variables taking **more than 15** different values)*.

We use the [`OneHotEncoder` class](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) from [scikit-learn](https://scikit-learn.org/stable/index.html) to get one-hot encodings. There are a number of parameters that can be used to customize its behavior.

used parameters:
- `handle_unknown='ignore'` to avoid errors when the validation data contains classes that aren't represented in the training data, and
- `sparse=False` ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix).

To use the encoder, we supply only the categorical columns that we want to be one-hot encoded. For instance, to encode the training data, we supply `X_train[object_cols]`. `object_cols` in the code cell below is a list of the column names with categorical data, and so `X_train[object_cols]` contains all of the categorical data in the training set.)

```Python
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
```
```
Out:
MAE from Approach 3 (One-Hot Encoding):
166089.4893009678
```

## Pipelines
A critical skill for deploying (and even testing) complex models with pre-processing.



