## Real / Fake News Classification using scikit-learn

## Method

- [load_data()](#load_data) : Load data and split it into train, validation, and test sets
- [select_model()](#select_model) : Select the best model using DecisionTreeClassifier with different criterion and max_depth
- [compute_information_gain()](#compute_information_gain) : Compute information gain for the top three keywords of the best model

### `load_data()`

1. Load data from **clean_fake.txt** and **clean_real.txt**

```python
with open('clean_fake.txt', 'r') as f:
    fake_news = f.read().splitlines() 
with open('clean_real.txt', 'r') as f:
    real_news = f.read().splitlines()
```

2. Data Preprocessing with **CountVectorizer** (example)

|             | the |donald|trump|...|     Label    |
|-------------|:---:|:----:|:---:|---|--------------|
|**headline1**|  0  |  1   |  1  |   | **1 (Real)** |
|**headline2**|  3  |  0   |  2  |   | **0 (Fake)** |
|**headline3**|  1  |  2   |  1  |   | **1 (Real)** |
|...|...|...|...||...|

3. Splitting the dataset into **Training(70%), Validation(15%), Test(15%)** sets

```python
# Total(100%) -->  Train(70%) | Temp(30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3)
# Temp(30%)   -->  Valid(15%) | Test(15%)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size = 0.5)
```

### `select_model()`


### `compute_information_gain()`

