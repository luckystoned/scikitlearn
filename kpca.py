import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart.head(5))

    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    dt_features = StandardScaler().fit_transform(dt_features)

    x_train, x_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    print(x_train.shape)    
    print(y_train.shape)

    kpca = KernelPCA(n_components=4, kernel='poly')
    kpca.fit(x_train)
    plt.scatter(dt_train[:, 0], dt_train[:, 1], c=y_train, cmap='viridis')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('KPCA')
    plt.show()
     

    dt_train = kpca.transform(x_train)
    dt_test = kpca.transform(x_test)

    logistic = LogisticRegression(solver='lbfgs')
    logistic.fit(dt_train, y_train)

    print("Score KPCA: ", logistic.score(dt_test, y_test))


    
