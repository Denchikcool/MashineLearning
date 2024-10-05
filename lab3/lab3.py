import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    data_X = preprocessing.normalize(data[:, :-1])
    data_Y = data[:, -1]
    train_x, test_x, train_y, test_y = train_test_split(data_X, data_Y, test_size=0.3, stratify=data_Y)
    return train_x, test_x, train_y, test_y

def train_linear_regression(train_x, train_y):
    linear_regression = LinearRegression()
    linear_regression.fit(train_x, train_y)
    return linear_regression

def evaluate_model(model, test_x, test_y):
    predicted = model.predict(test_x)
    success = 0
    for i in range(len(test_x)):
        if abs(test_y[i] - predicted[i]) < 1:
            success += 1
    return success / len(test_x) * 100

def run_evaluation(data, n_iterations):
    total_accuracy = 0
    for i in range(n_iterations):
        train_x, test_x, train_y, test_y = preprocess_data(data)
        model = train_linear_regression(train_x, train_y)
        accuracy = evaluate_model(model, test_x, test_y)
        print(f"{i + 1}) Точность:  {accuracy :.3f}%")
        total_accuracy += accuracy
    return total_accuracy / n_iterations

def main():
    file_name = "winequalityN.csv"
    n_iterations = 30

    data = pd.read_csv(file_name, header=0).fillna(0)
    data.loc[data.type == 'white', 'type'] = 0
    data.loc[data.type == 'red', 'type'] = 1
    data = data.to_numpy()

    print(f'\nВсе вина:')
    average_accuracy = run_evaluation(data, n_iterations)
    print(f'Средняя точность: {average_accuracy:.3f}% за {n_iterations} проходов\n\n')

    print(f'Белые вина:')
    white_data = data[data[:, 0] == 0]
    average_accuracy = run_evaluation(white_data, n_iterations)
    print(f'Средняя точность: {average_accuracy:.3f}% за {n_iterations} проходов\n\n')

    print(f'Красные вина:')
    red_data = data[data[:, 0] == 1]
    average_accuracy = run_evaluation(red_data, n_iterations)
    print(f'Средняя точность: {average_accuracy:.3f}% за {n_iterations} проходов')

if __name__ == "__main__":
    main()