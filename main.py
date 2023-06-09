from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt
import sys

class ODESolver:
    def __init__(self, f, y0, x0, xn, h, eps):
        self.f = f
        self.y0 = y0
        self.x0 = x0
        self.xn = xn
        self.h = float(h)
        self.eps = eps


    def exact_solution_1(self):
        def solution_1(x, C):
            return -(x) - 1 + C * np.exp(x)

        C = self.y0 + self.x0 + 1
        x = np.linspace(self.x0, self.xn, 100000)
        y = solution_1(x, C)
        return x[:100000], y

    def exact_solution_2(self):
        def solution_2(x, C):
            return np.sin(x) / 2 - np.cos(x) / 2 + C / np.exp(x)

        C = np.exp(self.x0)*(self.y0 - np.sin(self.x0) / 2 + np.cos(self.x0) / 2)
        x = np.linspace(self.x0, self.xn, 100000)
        y = solution_2(x, C)
        return x[:100000], y

    def exact_solution_3(self):
        def solution_3(x, C):
            return x ** 4 / 4 + C

        C = self.y0 - self.x0 ** 4 / 4
        x = np.linspace(self.x0, self.xn, 100000)
        y = solution_3(x, C)
        return x[:100000], y



    def euler(self):
        def cycle(y0, i, counter):
            for j in np.arange(i, i + self.h + 0.0000000001, self.h / counter):
                y2 = y0 + (self.h / counter) * self.f(j, y0)
                if abs(j-i-self.h) <= 0.001:
                    return y0
                y0 = y2

        integer = 0
        results = []
        table = [['№', 'x_i', 'y_i', 'f(x_i, y_i)', 'y_i(2)', 'f(x_i, y_i(2))', 'Деление шага']]
        table = PrettyTable(table[0])
        popper = 0
        R = 0
        x0 = self.x0
        y0 = self.y0
        y0_prev = self.y0
        p = 1
        n = int((self.xn + 0.001 - self.x0) / self.h)
        if n > 100000:
            print("Слишком большое n, выберете другие интервалы")
            sys.exit(0)
        for i in np.arange(self.x0, self.xn + 0.001, self.h):
            y1 = self.y0 + self.h * self.f(i, self.y0)
            y2 = self.y0
            counter = 2
            if i != x0:
                y2 = cycle(y0, i - self.h, counter)
                R = np.abs(y2 - y0) / (2 ** p - 1)
                while R > self.eps:
                    popper += 1
                    # if popper > 10:
                    #     print("С данными входными данными не получится найти ответ")
                    #     sys.exit(0)
                    counter = counter * 2
                    y2 = cycle(y0, i - self.h, counter)
                    R = np.abs(y2 - y0) / (2 ** counter - 1)
                y0 = y2

            row = [integer, round(i, 3), round(y2, 3), round(
                self.f(i, y2), 3), round(self.y0, 3), round(self.f(i, self.y0), 3), counter]
            table.add_row(row)
            results.append((i, y2))
            integer += 1
            self.y0 = y1
        print(table)
        self.y0 = y0_prev
        return results, R

    def getK(self, i, h, y0):
        k1 = h * self.f(i, y0)
        k2 = h * self.f(i + h / 2, y0 + k1 / 2)
        k3 = h * self.f(i + h / 2, y0 + k2 / 2)
        k4 = h * self.f(i + h, y0 + k3)
        return k1, k2, k3, k4

    def runge_kutta_4(self):
        def cycle(y0, i, counter):
            for j in np.arange(i, i + self.h + 0.0000001, self.h / counter):
                k1, k2, k3, k4 = self.getK(j, self.h / counter, y0)
                y2 = y0 + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                if abs(j-i-self.h) <= 0.001:
                    return y0
                y0 = y2

        integer = 0
        results = []
        result = []
        table = [['№', 'x_i', 'y_i', 'y_i/2', 'k1', 'k2', 'k3',
                  'k4', 'f(x_i, y_i)', 'Деление шага']]
        table = PrettyTable(table[0])
        x0 = self.x0
        y0 = self.y0
        y0_prev = self.y0
        p = 1
        n = int((self.xn + 0.001 - self.x0) / self.h)
        if n > 100000:
            print("Слишком большое n, выберете другие интервалы")
            sys.exit(0)
        for i in np.arange(self.x0, self.xn + 0.001, self.h):
            k1, k2, k3, k4 = self.getK(i, self.h, self.y0)

            y1 = self.y0 + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            y2 = self.y0
            counter = 1
            if i != x0:
                counter = 2
                y2 = cycle(y0, i - self.h, counter)
                R = np.abs(y2 - y0) / (2 ** p - 1)
                while R > self.eps:
                    counter = counter * 2
                    y2 = cycle(y0, i - self.h, counter)
                    R = np.abs(y2 - y0) / (2 ** counter - 1)
                y0 = y2
            p = max(p, counter)
            row = [integer, round(i, 3), round(y2, 3), round(self.y0, 3), round(k1, 3), round(k2, 3), round(k3, 3), round(k4, 3),
                   round(self.f(i, y2), 3), p]
            result.append((y2, self.f(i, y2)))
            results.append((i, y2))
            table.add_row(row)
            integer += 1
            self.y0 = y1
        print(table)
        self.y0 = y0_prev
        return result, results, R

    def miln(self, result):
        y = []
        y.append(result[0][0])
        y.append(result[1][0])
        y.append(result[2][0])
        y.append(result[3][0])

        f = []
        f.append(result[0][1])
        f.append(result[1][1])
        f.append(result[2][1])
        f.append(result[3][1])

        table = [['№', 'x_i', 'y_i', 'f(x_i, y_corr)']]
        table = PrettyTable(table[0])
        results = []
        for i in range(4):
            table.add_row([i, round(self.x0 + self.h * i, 3), round(y[i], 3), round(f[i], 3)])
            results.append((self.x0 + self.h * i, y[i]))
        counter = 4
        for i in np.arange(self.x0 + self.h * 4, self.xn + 0.001, self.h):
            y_prog = y[counter - 4] + 4 * self.h / 3 * (2 * f[counter-3] - f[counter-2] + 2 * f[counter - 1])
            f_prog = self.f(i, y_prog)
            y_corr = y[counter - 2] + self.h / 3 * (f[counter - 2] + 4 * f[counter - 1] + f_prog)
            f_corr = self.f(i, y_corr)
            if (i != self.x0 + self.h * 4):
                while (abs(y_corr - y_prog) >= self.eps):
                    y_prog = y_corr
                    f_prog = self.f(i, y_prog)
                    y_corr = y[counter - 2] + self.h / 3 * (f[counter - 2] + 4 * f[counter - 1] + f_prog)
                    f_corr = self.f(i, y_corr)
            y.append(y_corr)
            f.append(f_corr)
            results.append((i, y_corr))
            row = [counter, round(i, 3), round(y_corr, 3), round(f_corr, 3)]
            counter += 1
            table.add_row(row)
        print(table)
        return results


def main():
    f1 = lambda x, y: x + y
    f2 = lambda x, y: np.sin(x) - y
    f3 = lambda x, y: x ** 3

    print("f1 = x + y")
    print("f2 = sin(x) - y")
    print("f3 = x^3")
    func_input = input("Выберите функцию f1/f2/f3 ")

    func = f1

    match func_input:
        case "f1":
            func = f1
        case "f2":
            func = f2
        case "f3":
            func = f3
        case _:
            main()
            return

    y0 = float(input("Введите начальное условие y0 = y(x0): "))
    x0 = float(input("Введите начало интервала дифференцирования x0: "))
    xn = float(input("Введите конец интервала дифференцирования xn: "))
    h = float(input("Введите шаг h: "))
    epsilon = float(input("Введите погрешность eps:"))
    solver = ODESolver(func, y0, x0, xn, h, epsilon)

    if func == f3:
        x, y = solver.exact_solution_3()
    elif func == f2:
        x, y = solver.exact_solution_2()
    else:
        x, y = solver.exact_solution_1()

    plt.plot(x, y, label='Exact solution')



    print("Эйлер: ")
    dotsEuler, R = solver.euler()
    print(f"Точность метода по правилу Рунге:{R}")
    print("Рунге-Кутт: ")
    result, dotsRunge_kutt, R = solver.runge_kutta_4()
    print(f"Точность метода по правилу Рунге:{R}")
    if len(result)<4:
        print("Точек меньше чем 4, метод не отработал")
        dotsMiln = None
    else:
        print("Милн: ")
        dotsMiln = solver.miln(result)
    for i in range(len(dotsEuler)):
        plt.plot(dotsEuler[i][0], dotsEuler[i][1], "o", color='pink')
        plt.plot(dotsRunge_kutt[i][0], dotsRunge_kutt[i][1], "o", color='yellow')
        if (not dotsMiln is None):
            plt.plot(dotsMiln[i][0], dotsMiln[i][1], "o", color='green')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()

