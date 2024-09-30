from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

def train_and_evaluate_model(file_name, test_size=0.3, n_splits=30):
    data = np.genfromtxt(file_name, delimiter=',', skip_header=True)

    X = data[:, :-1]
    y = data[:, -1]

    best_params = find_best_params(X, y, test_size)
    print(f"Лучшими параметрами получилось:\nГлубина = {best_params["max_depth"]}\nМинимальное количество ветвей = {best_params["min_samples_leaf"]}")

    average_accuracy = cross_validate_model(X, y, best_params, test_size, n_splits)

    return average_accuracy

def find_best_params(X, y, test_size):
    best_accuracy = 0
    best_params = {}

    for max_depth in range(2, 20):
        for min_samples_leaf in range(2, 20):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
            for col in range(X_train.shape[1]):
                if "?" in X_train[:, col]:
                    X_train[:, col] = pd.to_numeric(X_train[:, col], errors='coerce') 
                    X_test[:, col] = pd.to_numeric(X_test[:, col], errors='coerce')
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            X_train = imp.fit_transform(X_train)
            X_test = imp.transform(X_test)

            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}

    return best_params

def cross_validate_model(X, y, params, test_size, n_splits):
    total_accuracy = 0

    for i in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        for col in range(X_train.shape[1]):
            if "?" in X_train[:, col]:
                X_train[:, col] = pd.to_numeric(X_train[:, col], errors='coerce') 
                X_test[:, col] = pd.to_numeric(X_test[:, col], errors='coerce')
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        X_train = imp.fit_transform(X_train)
        X_test = imp.transform(X_test)

        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        total_accuracy += accuracy

        print(f"{i+1}. Точность:  {(accuracy * 100):.3f}%")

    average_accuracy = total_accuracy / n_splits
    print(f"Средняя точность: {(average_accuracy * 100):.3f}% за {n_splits} разбиений")

    return average_accuracy

if __name__ == "__main__":
    file_name = "heart_data.csv"
    average_accuracy = train_and_evaluate_model(file_name)
