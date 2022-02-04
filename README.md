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
> A critical skill for deploying (and even testing) complex models with pre-processing.
> 
> full example of pipelines: [kaggle.com/alexisbcook/exercise-pipelines](https://www.kaggle.com/alexisbcook/exercise-pipelines); or [exercise-pipelines.ipynb]()

A simple way to keep your data preprocessing and modeling code organized. Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.

***Why to use pipelines: ***
1. **Cleaner Code:** Accounting for data at each step of preprocessing can get messy. With a pipeline, you won't need to manually keep track of your training and validation data at each step. like on [DAGsHub](https://dagshub.com/)
2. **Fewer Bugs:** There are fewer opportunities to misapply a step or forget a preprocessing step.
3. **Easier to Productionize:** It can be surprisingly hard to transition a model from a prototype to something deployable at scale. We won't go into the many related concerns here, but pipelines can help.

We take a peek at the training data from [Melbourne Housing dataset](https://www.kaggle.com/dansbecker/melbourne-housing-snapshot/home) with the `head()` method below. Notice that the data contains both [categorical data](https://github.com/gabboraron/Intermediate-Machine-Learning-Kaggle/edit/main/README.md#categorical-variables) and columns with [missing values](https://github.com/gabboraron/Intermediate-Machine-Learning-Kaggle/edit/main/README.md#missing-values). With a pipeline, it's easy to deal with both!
```
       Type   Method  Regionname            Rooms   Distance  Postcode  Bedroom2  Bathroom  Car   Landsize  BuildingArea  YearBuilt   Lattitude   Longtitude  Propertycount
12167   u       S     Southern Metropolitan   1       5.0     3182.0      1.0        1.0    1.0   0.0         NaN          1940.0     -37.85984   144.9867     13240.0
6524    h       SA    Western Metropolitan    2       8.0     3016.0      2.0        2.0    1.0   193.0       NaN          NaN        -37.85800   144.9005     6380.0
8413    h       S     Western Metropolitan    3      12.6     3020.0      3.0        1.0    1.0   555.0       NaN          NaN        -37.79880   144.8220     3755.0
2919    u       SP    Northern Metropolitan   3      13.0     3046.0      3.0        1.0    1.0   265.0       NaN          1995.0     -37.70830   144.9158     8870.0
6043    h       S     Western Metropolitan    3      13.3     3020.0      3.0        1.0    2.0   673.0       673.0        1970.0     -37.76230   144.8272     4217.0
```

### Step 1: Define Preprocessing Steps
We use [sklearn.compose.ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) to boundle dfferent preprocessing steps:
- impute missing values in numerical data
- impute missing values and applies one-hot encoding in categorial data

```Python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```

### Step 2: Define the Model
Using [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
```Python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
```

### Step 3: Create and Evaluate the Pipeline
```Python
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)    # the pipeline automatically preprocesses the features
                                        # before generating predictions. 
                                        # (However, without a pipeline, we have to remember 
                                        # to preprocess the validation data before making predictions.)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```
```
Out:
MAE: 160679.18917034855
```

> ***NOTE: If MAE is too high amend `numerical_transformer`, `categorical_transformer`, and/or `model` to get better performance.***
> 
> *In the case of `numerical_transformer` and `categorical_transformer` this means that you have to change the [`SimpleImputer()`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)'s `strategy` attribute  and/or set up better `fill_value` too.*
> 
> *In the case of `model` you can change in the [`RandomForestRegressor()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) 's `n_estimators` and `random_state`.* 

## Cross-Validation
[Wikipedia](https://en.wikipedia.org/wiki/Cross-validation_(statistics))

> A better way to test your models.
>
> You will face choices about what predictive variables to use, what types of models to use, what arguments to supply to those models, etc. So far, you have made these choices in a data-driven way by measuring model quality with a validation (or holdout) set.
>
> *ex: imagine you have a dataset with 5000 rows. You will typically keep about 20% of the data as a validation dataset, or 1000 rows. But this leaves some random chance in determining model scores. That is, a model might do well on one set of 1000 rows, even if it would be inaccurate on a different 1000 rows*
>
> ***Unfortunately, we can only get a large validation set by removing rows from our training data, and smaller training datasets mean worse models!***
> 
> ![3.1. Cross-validation: evaluating estimator performance - scikit-learn](https://scikit-learn.org/stable/_images/grid_search_workflow.png)

We could begin by dividing the data into 5 pieces, each 20% of the full dataset. In this case, we say that we have broken the data into 5 *"folds"*.

- **Experiment 1**, we use the first fold as a validation (or holdout) set and everything else as training data. This gives us a measure of model quality based on a 20% holdout set.
- **Experiment 2**, we hold out data from the second fold (and use everything except the second fold for training the model). The holdout set is then used to get a second estimate of model quality.
- We repeat this process, using every fold once as the holdout set. Putting this together, 100% of the data is used as holdout at some point, and we end up with a measure of model quality that is based on all of the rows in the dataset (even if we don't use all rows simultaneously).

![wikipedia - Illustration of leave-one-out cross-validation (LOOCV) when n = 8 observations. A total of 8 models will be trained and tested.](https://upload.wikimedia.org/wikipedia/commons/c/c7/LOOCV.gif)

And we can do this on the all dataset:
![grid_search_cross_validation](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)

***it can take longer to run, because it estimates multiple models (one for each fold)***

**when use it:**
- *For small datasets*, where extra computational burden isn't a big deal, you should run cross-validation. If your model takes a couple minutes or less to run, it's probably worth switching to cross-validation.
- *For larger datasets*, a single validation set is sufficient. Your code will run faster, and you may have enough data that there's little need to re-use some of it for holdout.

We obtain the cross-validation scores with the [`cross_val_score()`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html) function from [scikit-learn](https://scikit-learn.org/stable/index.html).
- `cv` - Determines the cross-validation splitting strategy. Possible inputs for cv are:
   - `None` - by default 5-fold cross validation
   - `int` -  to specify the number of folds in a (Stratified)KFold
   - [CV splitter](https://scikit-learn.org/stable/glossary.html#term-CV-splitter) like [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split): `clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)`
     - `cross-validation generator` - A non-estimator family of classes used to split a dataset into a sequence of train and test portions 
     -  `cross-validation estimator` An estimator that has built-in cross-validation capabilities to automatically select the best hyper-parameters ex: [`ElasticNetCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV), [`LogisticRegressionCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV)
     -  `scorer` - A non-estimator callable object which evaluates an estimator on given test data, returning a number. `clf.score(X_test, y_test)`. **higher return values are better than lower return values**. If you are using [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) or [`cross_val_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score) you can set [`scoring`](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) parameter.
   - An iterable that generates (train, test) splits as arrays of indices, like [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)

```Python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error') # which is a regressor: 
                                                                 # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error

print("MAE scores:\n", scores)
print("\nAverage MAE score (across experiments):")
print(scores.mean())
```
```
Out:
MAE scores:
 [301628.7893587  303164.4782723  287298.331666   236061.84754543
 260383.45111427]
Average MAE score (across experiments):
277707.3795913405
```

***Using cross-validation yields a much better measure of model quality, with the added benefit of cleaning up our code: note that we no longer need to keep track of separate training and validation sets. So, especially for small datasets, it's a good improvement!***





