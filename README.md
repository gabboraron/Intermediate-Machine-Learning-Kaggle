# Intermediate-Machine-Learning-Kaggle
Handle missing values, non-numeric values, data leakage, and more.

Introduction in [exercise-introduction.ipynb](https://github.com/gabboraron/Intermediate-Machine-Learning-Kaggle/blob/main/exercise-introduction.ipynb)

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
Out:
```
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
Out:
```
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

