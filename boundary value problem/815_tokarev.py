import numpy as np
import matplotlib.pyplot as plt

"""
    У заданного уравнения взяли производную сложной функции и потом работали с ними, 
    чтобы не мучаться с половинными номерами. 
    
    
"""

number_of_steps = 20
step_size = 1 / number_of_steps

x = np.linspace(0, 1, number_of_steps + 1)  # если стартуем с нулевой точки, то надо сделать 20 шагов


def g(x):
    return np.cos(4 * x) * np.cos(4 * x) / 4 + 1


def grad_g(x):
    return - np.sin(8 * x)


def p(x):
    return np.cos(4 * x) * np.cos(4 * x) + 1


def y(x):
    return np.cos(2 * x)


def grad_y(x):
    return - 2 * np.sin(2 * x)


def f(x):
    return 2 * np.sin(2 * x) * np.sin(8 * x) + 2 * np.cos(2 * x) * (np.cos(4 * x) * np.cos(4 * x) + 2.5)


class TridiagonalMatrix:
    def __init__(self, n):
        self.data = np.array((n + 3) * [np.zeros(4)])  # 3 диагонали и столбец свободных членов
        self.size = n
        self.__step_size = 1 / self.size

        self.data[0, 0] = - 1 / (2 * self.__step_size)
        self.data[0, 2] = 1 / (2 * self.__step_size)
        self.data[0, 3] = grad_y(0)

        self.data[-1, 0] = - 1 / (2 * self.__step_size)
        self.data[-1, 2] = 1 / (2 * self.__step_size)
        self.data[-1, 3] = grad_y(1)

        x_k = np.linspace(0, 1, self.size + 1)
        g_k = g(x_k)
        grad_g_k = grad_g(x_k)
        f_k = f(x_k)
        p_k = p(x_k)

        for i in range(1, self.size + 2):
            self.data[i, 0] = - g_k[i - 1] / (self.__step_size**2) - grad_g_k[i - 1] / (2 * self.__step_size)
            self.data[i, 1] = p_k[i - 1] + 2 * g_k[i - 1] / (self.__step_size**2)
            self.data[i, 2] = - g_k[i - 1] / (self.__step_size**2) + grad_g_k[i - 1] / (2 * self.__step_size)
            self.data[i, 3] = f_k[i - 1]

    def normalize_data(self):
        normal_data = np.array((self.size + 1) * [np.zeros(4)])
        normal_data[0] = self.data[1] - self.data[0] * (self.data[1, 0] / self.data[0, 0])
        normal_data[1:-1] = self.data[2:-2]
        normal_data[-1] = self.data[-2] - self.data[-1] * (self.data[-2, 2] / self.data[-1, 2])

        if normal_data[0, 0] == 0 and normal_data[-1, 2] == 0:
            print("Well normalize data)")

        self.data = normal_data

    def sweep_method(self):
        for i in range(1, self.size + 1):
            k = self.data[i, 0] / self.data[i - 1, 1]
            self.data[i, 0] = 0
            self.data[i, 1] += - self.data[i - 1, 2] * k
            self.data[i, 3] += - self.data[i - 1, 3] * k

        for i in range(self.size - 1, -1, -1):
            k = self.data[i, 2] / self.data[i + 1, 1]
            self.data[i, 2] = 0
            self.data[i, 1] += - self.data[i + 1, 0] * k
            self.data[i, 3] += - self.data[i + 1, 3] * k

    def solution(self):
        return self.data[:, 3] / self.data[:, 1]


matrix = TridiagonalMatrix(number_of_steps)
matrix.normalize_data()
matrix.sweep_method()
solution = matrix.solution()

accuracy_solution = y(x)

if len(accuracy_solution) != len(solution):
    print("ERROR! Not equal length of solutions!")
    print("AS = {}\t S = {}".format(len(accuracy_solution), len(solution)))


print("Вывод таблицы значений в узлах для n = 20")
print("{:^10} | {:^10}".format("AprxSol", "AccurSol"))
print("-"*23)

for i in range(len(x)):
    print("{:^10.4} | {:^10.4}".format(solution[i], accuracy_solution[i]))

norm = np.abs(accuracy_solution - solution)

print("CUBE: {}".format(np.max(norm)))
if np.max(norm) < 10**(-2):
    print("Убедились, что норма ошибки дает 2-ой порядок малости.")


plt.figure(figsize=[15, 15])
plt.subplot(2, 1, 1)
plt.title("Accuracy solution:")
plt.scatter(x,accuracy_solution)
plt.subplot(2, 1, 2)
plt.title("Metod\'s solution")
plt.scatter(x, solution)
# plt.show()  Использовалось ля отладки программы
plt.savefig("result.png")
