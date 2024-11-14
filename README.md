# Real / Fake News Classification using scikit-learn

- Learning how to use the DecisionTreeClassifier in scikit-learn by building a fake news classifier

## Method

- [load_data()](#load_data) : Load data and split it into train, validation, and test sets
- [select_model()](#select_model) : Select the best model using DecisionTreeClassifier with different criterion and max_depth
- [compute_information_gain()](#compute_information_gain) : Compute information gain for the top three keywords of the best model


### load_data()

1. Load data from **clean_fake.txt** and **clean_real.txt**

    ```python
    with open('clean_fake.txt', 'r') as f:
        fake_news = f.read().splitlines() 
    with open('clean_real.txt', 'r') as f:
        real_news = f.read().splitlines()
    ```
    

2. Data preprocessing with **CountVectorizer** 

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


### select_model()

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

### compute_information_gain()

1. Compute the **total entropy $`\ H(Y) `$**
   
    ### $`\ H(Y) = -\sum\limits_{y \in Y} P(y) log_2 P(y) `$

    ```python
    _, counts = np.unique(y, return_counts = True)
    probabilities = counts / len(y)
    total_entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    ```
    
2. Extract the **top three keywords**
   
    ```python
    top3 = np.argsort(best_model.feature_importances_)[-3:]
    ```

3. Compute the **conditional entropy $`\ H(Y|X^\left(i\right)) `$** and **information gain $`\ IG(Y, X^\left(i\right)) `$** for each keyword

    ### $`\ H(Y|X^\left(i\right)) = \sum\limits_{x \in X^\left(i\right)} P(x) H(Y|X^\left(i\right) = x)`$

    ### $`\ IG(Y, X^\left(i\right)) = H(Y) - H(Y|X^\left(i\right)) `$

    ```python
    for i in top3:
        feature_column = X[:, i]
        values, counts = np.unique(feature_column, return_counts=True)

        # H(Y|Xi)
        conditional_entropy = 0
        for value, count in zip(values, counts):
            index = np.where(feature_column == value)[0]
            y_subset = y[index]
            _, target_counts = np.unique(y_subset, return_counts=True)
            
            prob_y_given_x = target_counts / len(y_subset)
            entropy_y_given_x = -np.sum(prob_y_given_x * np.log2(prob_y_given_x))

            conditional_entropy += (count / len(feature_column)) * entropy_y_given_x

        # IG(Y, Xi) <- H(Y) - H(Y|Xi)
        info_gain_dict[vectorizer.get_feature_names_out()[i]] = total_entropy - conditional_entropy
    ```

- example (conditional entropy of the keyword)

    |   feature_column    | count |       y_subset        |
    |---------------------|:-----:|:---------------------:|
    |        **X1**       |   0   |      **1 (Real)**     |
    |        **X2**       |   0   |      **0 (Fake)**     |
    |        **X3**       |   0   |      **1 (Real)**     |
    |        **X4**       |   1   |      **1 (Real)**     |
    |        **X5**       |   1   |      **0 (Fake)**     |
    |        **X6**       |   2   |      **1 (Real)**     |
    |        **X7**       |   2   |      **0 (Fake)**     |
    |         ...         |  ...  |           ...         |

    1. conditional entropy of **count = 0**:
       $`\ - \left(\frac{2}{3}\right) log_2 \left(\frac{2}{3}\right) - \left(\frac{1}{3}\right) log_2 \left(\frac{1}{3}\right) `$

    2. conditional entropy of **count = 1**:
       $`\ - \left(\frac{1}{2}\right) log_2 \left(\frac{1}{2}\right) - \left(\frac{1}{2}\right) log_2 \left(\frac{1}{2}\right) `$
 
    3. conditional entropy of **count = 2**:
       $`\ - \left(\frac{1}{2}\right) log_2 \left(\frac{1}{2}\right) - \left(\frac{1}{2}\right) log_2 \left(\frac{1}{2}\right) `$
    
    4. ...

    
    - **$`\ H(Y|X^\left(i\right)) `$**
  
        $`\ = \left(\frac{3}{7}\right) \times \left\{ - \left(\frac{2}{3}\right) log_2 \left(\frac{2}{3}\right) - \left(\frac{1}{3}\right) log_2 \left(\frac{1}{3}\right)\right\} `$
    
        $`\ + \left(\frac{2}{7}\right) \times \left\{ - \left(\frac{1}{2}\right) log_2 \left(\frac{1}{2}\right) - \left(\frac{1}{2}\right) log_2 \left(\frac{1}{2}\right)\right\} `$
      
        $`\ + \left(\frac{2}{7}\right) \times \left\{ - \left(\frac{1}{2}\right) log_2 \left(\frac{1}{2}\right) - \left(\frac{1}{2}\right) log_2 \left(\frac{1}{2}\right)\right\} `$
      
        $`\ + `$ ... 

  
## Observation

- ### plot the accuracy for each criterion

<img width="700" alt="plot" src="https://github.com/user-attachments/assets/b9e9fc2d-80da-4974-8766-7f9d2e48c912">


- ### plot the decision tree of the best model with max_depth = 2

<img width="700" alt="tree" src="https://github.com/user-attachments/assets/0cd9b279-4b2b-4179-ad24-a6c87eafebd7">


