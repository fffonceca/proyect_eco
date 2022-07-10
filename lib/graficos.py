import numpy as np
import matplotlib.pyplot as plt


def grafico_estados(x_vector: np.array, y_vector: np.array, refs: np.array):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(1, 7):
        ax = fig.add_subplot(2, 3, i)


    plt.show()
