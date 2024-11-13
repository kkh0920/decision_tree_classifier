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
    

2. Data Preprocessing with **CountVectorizer** 

    ```python
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(fake_news + real_news)
    y = np.array([0] * len(fake_news) + [1] * len(real_news))
    ```

- example
      
    |             | the |donald|trump|...|       Y        |
    |-------------|:---:|:----:|:---:|---|:--------------:|
    |    **X1**   |  0  |  1   |  1  |   |  **1 (Real)**  |
    |    **X2**   |  3  |  0   |  2  |   |  **0 (Fake)**  |
    |    **X3**   |  1  |  2   |  1  |   |  **1 (Real)**  |
    |     ...     | ... |  ... | ... |   |      ...       |


3. Splitting the dataset into **Training(70%), Validation(15%), Test(15%)** sets

    ```python
    # Total(100%) -->  Train(70%) | Temp(30%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3)
    # Temp(30%)   -->  Valid(15%) | Test(15%)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size = 0.5)
    ```


### `select_model()`

- Select the **Decision Tree Classifier** with the **best accuracy in the validation set**
- Criterion: **entropy, log_loss, gini**
- Tree Depth: 1 ~ **depth**

    ```python
    depth = 150
    
    best_score = 0
    best_model = None

    for criterion in ['entropy', 'log_loss', 'gini']:        
        for max_depth in range(1, depth + 1): 
            model = DecisionTreeClassifier(max_depth = max_depth, criterion = criterion)
            model.fit(X_train, y_train)
    
            y_pred = model.predict(X_valid)            
            score = accuracy_score(y_valid, y_pred)
    
            if score > best_score:
                best_score = score
                best_model = model
    ```

### `compute_information_gain()`


