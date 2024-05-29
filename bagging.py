import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    dt_hearth = pd.read_csv('./data/heart.csv')
    # print(dt_hearth['target'].describe())

    X = dt_hearth.drop(['target'], axis=1)
    y = dt_hearth['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print("="*64)
    print(accuracy_score(knn_pred, y_test))

    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)
    print("="*64)
    print(accuracy_score(bag_pred, y_test))

    knn_reg = KNeighborsRegressor().fit(X_train, y_train)
    knn_reg_pred = knn_reg.predict(X_test)
    print("="*64)
    print(knn_reg_pred)

    bag_reg = BaggingRegressor(base_estimator=KNeighborsRegressor(), n_estimators=50).fit(X_train, y_train)
    bag_reg_pred = bag_reg.predict(X_test)
    print("="*64)
    print(bag_reg_pred)
    print("="*64)