import matplotlib.pyplot as plt
import numpy as np


def grafico_estados(t: np.array, eX_vector: np.array, eY_vector: np.array):
    plt.plot(t, eX_vector, label="error x")
    plt.plot(t, eY_vector, label="error y")
    plt.xlabel("Tiempo (segs)")
    plt.ylabel("Error (mts)")
    plt.title("Error en distancias")
    plt.legend()
    plt.show()


def grafico_peso(t: np.array, reward_vector: np.array, torques_vector: np.array):
    plt.plot(t, reward_vector, label="recompensa")
    plt.plot(t, torques_vector, label="energía")
    plt.xlabel("Tiempo (segs)")
    plt.ylabel("Función de costos (mts)")
    plt.title("Dinámica de costo")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    t = np.linspace(0, 10, 3)
    ex = np.array([0.01, 0.11, -0.01])
    ey = np.array([0.01, 0.11, -0.01])*2

    grafico_estados(t, ex, ey)
