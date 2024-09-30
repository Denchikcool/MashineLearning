import pandas as pd
import numpy as np
import math
import random
import ast

def split_csv(input_file, testing_file, learning_file):
    # Читаем данные из CSV файла, игнорируя первую строку
    data = pd.read_csv(input_file)

    # Разделяем данные на тестовые и обучающие
    num_testing_rows = round(len(data) / 3)  # Берём 1/3 строк для тестирования
    testing_data = data.iloc[:num_testing_rows]
    learning_data = data.iloc[num_testing_rows:]

    # Сохраняем данные в отдельные CSV файлы
    testing_data.to_csv(testing_file, index=False, header=False)
    learning_data.to_csv(learning_file, index=False, header=False)

def sort_learning_data(learning_file):
    # Читаем данные из learning.csv
    data = pd.read_csv(learning_file, header=None)

    # Сортировка по первому и второму столбцу
    sorted_data = data.sort_values(by=[data.columns[0], data.columns[1]], ignore_index=True)

    # Сохранение отсортированных данных обратно в learning.csv
    sorted_data.to_csv(learning_file, index=False, header=False)

def calculate_distances_index_weight(learning_file):
    # Читаем данные из learning.csv
    data = pd.read_csv(learning_file, header=None)

    #X = data.iloc[:, 0].values
    #Y = data.iloc[:, 1].values
    # Создаем массив для хранения расстояний
    distances = []
    #distances = np.zeros((len(data), len(data)))

    # Вычисляем расстояния
    for i in range(len(data)):
        point_distances = []
        #index = 1
        
        for j in range(len(data)):
            # Получаем координаты точек
            x1 = data.iloc[i, 0]
            y1 = data.iloc[i, 1]
            x2 = int(data.iloc[j, 0])
            y2 = int(data.iloc[j, 1])
            classes = int(data.iloc[j, 2])
            
            # Вычисляем расстояние
            distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            #if distance == 0:
                #continue
            #if index > 200:
                #point_distances.append((index, distance, 0))
            point_distances.append((x2, y2, distance, classes))
            #index += 1
            #point_distances[j] = distance
        sorted_point_distances = list(sorted(point_distances, key=lambda x: x[2]))
        for k, point_distance in enumerate(sorted_point_distances):
            x2, y2, distance, classes = point_distance
            index = k   # Индекс начинается с 1
            if distance == 0:  # Если расстояние равно 0, индекс остается 0
                index = 0
            sorted_point_distances[k] = [x2, y2, index, distance, classes] #убрал возведение в степень
        distances.append(sorted_point_distances)
    return distances

# def knn(distances):
#     # print(type(distances))
#     best_k = 0
#     best_q = 0
#     best_degree = 0
#     #temp = list(distances)
#     #print(temp)
#     for q in range(1, 5):
#         for k in range(1, len(distances)):
#             correct_answers = 0
#             predicted = -1
#             for i in range(len(distances)):
#                 accuracy_0 = 0
#                 accuracy_1 = 0
#                 if (k + 2) > len(distances): break
#                 for j in range(1, k + 2):
#                     distances[i][j].append(math.pow(q / 10, distances[i][j][2]))
#                     if(distances[i][j][5] == 0):
#                         accuracy_0 += distances[i][j][4]
#                     else:
#                         accuracy_1 += distances[i][j][4]
#                 if accuracy_0 > accuracy_1:
#                     predicted = 0
#                 elif accuracy_0 < accuracy_1:
#                     predicted = 1

#                 if predicted == distances[i][0][4]:
#                     correct_answers += 1
#             accurany = (correct_answers / len(distances)) * 100
#             if best_degree < accurany:
#                 best_degree = accurany
#                 best_k = k
#                 best_q = q / 10
#                 print(f"{best_degree} {best_k} {best_q}")
#     print(best_k)
#     print(best_q)
#     print(best_degree)
#     return best_k, best_q

def knn(distances):
    best_k = 0
    best_q = 0
    best_degree = 0
    print(type(distances))

    for q in range(1, 5):  # Перебирайте k
        for k in range(1, len(distances)):  # Перебирайте q
            correct_answers = 0
            predicted = -1

            for i in range(len(distances)):  # Проходите по данным
                accuracy_0 = 0
                accuracy_1 = 0

                if (k + 2) > len(distances):
                    break  # Проверяйте, чтобы k не выходил за границы данных

                for j in range(1, k + 2):  # Проходите по k ближайшим соседям
                    # Применяйте взвешивание по q
                    distances[i][j].append(math.pow(q / 10, distances[i][j][2]))

                    if distances[i][j][4] == 0:
                        accuracy_0 += distances[i][j][5]
                    else:
                        accuracy_1 += distances[i][j][5]

                if accuracy_0 > accuracy_1:
                    predicted = 0
                elif accuracy_0 < accuracy_1:
                    predicted = 1

                if predicted == distances[i][0][4]:
                    correct_answers += 1

            accurany = (correct_answers / len(distances)) * 100

            # Обновляйте значения best_degree, best_k и best_q, если 
            # текущая точность лучше
            if best_degree < accurany:
                best_degree = accurany
                best_k = k
                best_q = q / 10
                print(f"{best_degree} {best_k} {best_q}")

    print(best_k)
    print(best_q)
    print(best_degree)
    return best_k, best_q

def weighted_knn(distances, target_index):

    # Извлекаем информацию о целевой точке
    # Класс целевой точки

    # Проверяем разные значения k от 1 до половины количества точек
    best_ks = []
    for i in range(len(distances)): #цикл идет по строчкам в csv (либо же по подмассивам)
        target_class = distances[i][0][5] #содержит класс точки в себе
        best_k = 1
        best_accuracy = 0
        for k in range(1, len(distances[target_index]) // 2 + 1):
            # Выбираем k ближайших соседей
            neighbors = distances[i][:k]
    
            # Подсчитываем взвешенную сумму классов соседей
            weighted_sum = 0
            for neighbor in neighbors:
                weighted_sum += neighbor[5] * neighbor[4]
                dotclass = neighbor[5]
    
            # Предсказываем класс целевой точки
            predicted_class = 1 if weighted_sum > 0.5 else 0
    
            # Вычисляем точность предсказания
            accuracy = 1 if predicted_class == dotclass else 0
    
            # Обновляем best_k и max_accuracy, если точность выше
            if accuracy > best_accuracy:
                best_k = k
                best_accuracy = accuracy
        best_ks.append(best_k)
    
    # Возвращаем k с максимальной точностью
    print(best_ks)
    return max(set(best_ks), key=best_ks.count)


def save_distances_to_csv(distances, output_file):
    # Создаем DataFrame из массива расстояний
    distances_df = pd.DataFrame(distances)

    # Сохраняем DataFrame в файл CSV
    distances_df.to_csv(output_file, index=False, header=False, sep=",")

def main():
    # Пути к файлам
    input_file = "data5.csv"
    testing_file = "testing.csv"
    learning_file = "learning.csv"
    output_file = "output.csv"
    #q = round(random.uniform(0.01, 0.99), 2)

    # Разделение CSV файла
    #split_csv(input_file, testing_file, learning_file)
    #split_csv(input_file, testing_file1, learning_file)
    # Тренируемся на learning.csv
    #sort_learning_data(learning_file)
    #distances = calculate_distances_index_weight(learning_file)
    #print(distances)
    
    #sort_learning_data(testing_file1)

    distances = calculate_distances_index_weight(testing_file)
    #save_distances_to_csv(distances, output_file)

    #distances = ast.literal_eval(distances)

    target_index = 0
    k, q = knn(distances)
    print(f"Оптимальное значение в обучающем наборе k = {k} при q = {q}")
    #save_distances_to_csv(distances, output_file)

if __name__ == '__main__':
    main()
