import numpy as np
import matplotlib.pyplot as plt

"""
    Рассматриваем решение уравнения вида:   l*u'' + 2*Lambda*u' + g*sin(u(t)) = 0 
                                            u(0) = u0, u'(0) = v0
                                            
    Преобразуем к виду: { u' = v, u(0) = u0, v(0) = v0
                        { v' + 2*L/l*v + g/l*sin(u) = 0
                               
    Введя переобозначение, имеем: gamma = L/l, omega^2 = g/l
                                            
    T ~= 2*pi*sqrt(l/g) и 10-20 шагов на период
    
    Решить используя схему Эйлера (ЕЕ) и метод Ранге-Кутта 4-порядка (RK)
    Также вывести точное решение, которое мы знаем (AN)
        
    Выводить таблицу: tn, un(EE), un(RK), un(AN)
    
    stdin: l, lambda, g, u0, v0
    stdout: tn, un(EE), un(RK), un(AN)
    
    Добавить вывод красивой информации в виде графика
"""


class Pendulum:
    # Мы можем задать количество точек, на которое хотим разбить период
    __n = 20

    def __init__(self, l, Lambda, g, u_0, v_0):
        self.__length = l
        self.__Lambda = Lambda
        self.__gravity_acceleration = g
        self.__u_0 = u_0
        self.__v_0 = v_0

        self.gamma = self.__Lambda / self.__length
        self.omega = np.sqrt(self.__gravity_acceleration / self.__length)
        self.T = 2 * np.pi * np.sqrt(l / g)
        self.tau = self.T / Pendulum.__n

        self.f_v = lambda v, u: -(2 * self.gamma * v + self.omega * self.omega * np.sin(u))
        self.f_u = lambda v, u: v

    def get_n(self):
        return self.__n

    def t_n(self):
        return np.array([i * self.tau for i in range(3 * Pendulum.__n + 1)])

    def euler_method(self):
        u_n = self.__u_0
        v_n = self.__v_0
        res = np.array([u_n])

        for _ in range(3 * Pendulum.__n):
            v_n = v_n + self.tau * self.f_v(v_n, u_n)
            u_n = u_n + self.tau * self.f_u(v_n, u_n)
            res = np.append(res, u_n)

        return res

    def rungekutta(self):
        u_n = self.__u_0
        v_n = self.__v_0
        res = np.array([u_n])

        for _ in range(3 * Pendulum.__n):
            # Аналог значений в матрице Бутчера для вычисления значений K
            A = np.array([0, 1/2, 1/2, 1])
            K_v = np.zeros(4)
            K_u = np.zeros(4)

            # Для уменьшения кода получилось создать в цикле вычисление K.
            # Но оставил вычисление по обычному, если будет когда-либо читать не знающий метода Ранге-Кутта пользователь

            for i in range(4):
                K_v[i] = self.f_v(v_n + self.tau * K_v[i - 1] * A[i], u_n + self.tau * K_u[i - 1] * A[i])
                K_u[i] = self.f_u(v_n + self.tau * K_v[i - 1] * A[i], u_n + self.tau * K_u[i - 1] * A[i])

            # K1_v = self.f_v(v_n, u_n)
            # K1_u = self.f_u(v_n, u_n)
            #
            # K2_v = self.f_v(v_n + self.tau * K1_v / 2, u_n + self.tau * K1_u / 2)
            # K2_u = self.f_u(v_n + self.tau * K1_v / 2, u_n + self.tau * K1_u / 2)
            #
            # K3_v = self.f_v(v_n + self.tau * K2_v / 2, u_n + self.tau * K2_u / 2)
            # K3_u = self.f_u(v_n + self.tau * K2_v / 2, u_n + self.tau * K2_u / 2)
            #
            # K4_v = self.f_v(v_n + self.tau * K3_v, u_n + self.tau * K3_u)
            # K4_u = self.f_u(v_n + self.tau * K3_v, u_n + self.tau * K3_u)

            v_n += self.tau * (K_v[0] / 6 + K_v[1] / 3 + K_v[2] / 3 + K_v[3] / 6)
            u_n += self.tau * (K_u[0] / 6 + K_u[1] / 3 + K_u[2] / 3 + K_u[3] / 6)
            res = np.append(res, u_n)

        return res

    def exact_solution(self):
        if self.gamma < self.omega:
            Omega = np.sqrt(self.omega ** 2 - self.gamma ** 2)
            C_1 = self.__u_0
            C_2 = (self.__v_0 + self.__u_0 * self.gamma) / Omega

            res = np.array([(C_1 * np.cos(Omega * i * self.tau) + C_2 * np.sin(Omega * i * self.tau))
                            * np.exp(-self.gamma * i * self.tau) for i in range(3 * Pendulum.__n + 1)])

        elif self.gamma > self.omega:
            Gamma = np.sqrt(self.gamma ** 2 - self.omega ** 2)
            C_1 = (-self.__v_0 + (Gamma - self.gamma) * self.__u_0) / 2 / Gamma
            C_2 = (self.omega ** 2 + self.__u_0(self.gamma + Gamma)) / 2 / Gamma

            res = np.array([(C_1 * np.exp(-Gamma * i * self.tau) + C_2 * np.exp(Gamma * i * self.tau))
                            * np.exp(-self.gamma * i * self.tau) for i in range(3 * Pendulum.__n + 1)])

        else:
            C_1 = self.__v_0 + self.__u_0 * self.gamma
            C_2 = self.__u_0

            res = np.array([(C_1 * i * self.tau + C_2)
                            * np.exp(-self.gamma * i * self.tau) for i in range(3 * Pendulum.__n + 1)])

        return res


l, Lambda, g, u_0, v_0 = map(float, input("Введите параметры уравнения и начальные условия для задачи Коши\n" +
                                          "А именно:\nlength, Lambda, gravity acceleration, the initial values (u, v)\n").split())

curr_pendulum = Pendulum(l, Lambda, g, u_0, v_0)

t_n = curr_pendulum.t_n()
EE = curr_pendulum.euler_method()
RK = curr_pendulum.rungekutta()
AN = curr_pendulum.exact_solution()

print("Вывод таблицы значений последовательностей решений для различных методов.")
print("{:^10} | {:^10} | {:^10} | {:^10}".format("t_n", "EE", "RK", "AN"))
print("-"*49)

for i in range(len(t_n)):
    print("{:^10.4} | {:^10.4} | {:^10.4} | {:^10.4}".format(t_n[i], EE[i], RK[i], AN[i]))


# Кусок кода ниже был необходим для визуализации решиний и приближений различными способами.

# plt.figure(figsize=[15, 15])
# plt.subplot(3, 1, 1)
# plt.title("Euler's metod")
# plt.scatter(t_n, EE)
# plt.subplot(3, 1, 2)
# plt.title("Runge-Kutta's metod")
# plt.scatter(t_n, RK)
# plt.subplot(3, 1, 3)
# plt.title("Accuracy soution")
# plt.scatter(t_n, AN)
# plt.show()
