from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

import numpy as np
import matplotlib.pyplot as plt

def load_data():
    # 1. 데이터를 읽어들이고,
    with open('clean_fake.txt', 'r') as f:
        fake_news = f.read().splitlines() 
    with open('clean_real.txt', 'r') as f:
        real_news = f.read().splitlines()
    
    # 2. CountVectorizer를 사용하여 데이터를 전처리하며,
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(fake_news + real_news)
    y = np.array([0] * len(fake_news) + [1] * len(real_news))
    
    # 3. 전체 데이터를 70%의 훈련 세트, 15%의 검증 세트, 15%의 테스트 세트로 무작위로 분할.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size = 0.5)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test, vectorizer

def select_model(X_train, X_valid, y_train, y_valid):
    depth = 150
    best_score = 0
    best_model = None
    plt.figure()
    # 분류 기준 : 'entropy', 'log_loss', 'gini'
    for criterion in ['entropy', 'log_loss', 'gini']:
        accuracy = []
        # 트리 깊이: 1 ~ depth
        for max_depth in range(1, depth + 1): 
            model = DecisionTreeClassifier(max_depth = max_depth, criterion = criterion)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)            
            score = accuracy_score(y_valid, y_pred)
            accuracy.append(score)
            # 가장 높은 성능의 모델 선택
            if score > best_score:
                best_score = score
                best_model = model
        # [Max Depth 대비 Accuracy] 플롯 추가 ('entropy', 'log_loss', 'gini')
        plt.plot(range(1, depth + 1), accuracy, label=criterion)
        
    plt.legend()
    plt.xlabel('Max depth')
    plt.ylabel('Accuracy')
    
    return best_model, best_score

def compute_information_gain(X_train, y_train, best_model):
    X = X_train.toarray()
    y = y_train
    
    # 전체 엔트로피 H(Y)
    _, counts = np.unique(y, return_counts = True)
    probabilities = counts / len(y)
    total_entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    info_gain_dict = {}
    # 가장 높은 성능의 분류기에서 최상위 세 가지 키워드 추출 
    top3 = np.argsort(best_model.feature_importances_)[-3:]
    for i in top3:
        feature_column = X[:, i]
        values, counts = np.unique(feature_column, return_counts=True)
        # 조건부 엔트로피 H(Y|Xi)
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
    
    return info_gain_dict

# main 함수
if __name__ == '__main__':
    print('Loading data...')
    X_train, X_valid, X_test, y_train, y_valid, y_test, vectorizer = load_data()
    print('Data loaded.\n')
    
    print('Selecting the best model...')
    best_model, best_score = select_model(X_train, X_valid, y_train, y_valid)
    print('Best model selected.\n')
    
    info_gain_dict = compute_information_gain(X_train, y_train, best_model)
    
    # [의사 결정 트리] 플롯 추가 (max_depth = 2)
    plt.figure()
    tree.plot_tree(best_model, feature_names = vectorizer.get_feature_names_out(), 
                    class_names = ['fake', 'real'], max_depth = 2, filled = True)
    
    print('------------------------------------')
    print('         Result of the model        ')
    print('------------------------------------')
    
    print('<Best Accuracy>', best_score)
    print('<Test Accuracy>', accuracy_score(y_test, best_model.predict(X_test)))
    
    i = 1
    print('------------------------------------')
    print('<Information Gain>')
    for key, value in info_gain_dict.items():
        print(f" {i}. {key} : {value}")
        i += 1
    print('------------------------------------\n')
    
    plt.show()