import matplotlib.pyplot as plt
import numpy as np
PID_PATH_EXPORT = "export/pid"
LIQ_PATH_EXPORT = "export/liq"
OSC_PATH_EXPORT = "export/osc"


def grafico_convergencia(t: np.array, e_vector: np.array):
    eX_vector = e_vector[:, 0]
    eY_vector = e_vector[:, 1]
    plt.plot(t, eX_vector, label="error x")
    plt.plot(t, eY_vector, label="error y")
    plt.xlabel("Tiempo (segs)")
    plt.ylabel("Error (mts)")
    plt.title("Error en distancias")
    plt.legend()
    plt.show()


def grafico_indices(t: np.array, ise_v: np.array, itse_v: np.array,
                    iae_v: np.array, itae_v: np.array):
    fig, axs = plt.subplots(2)
    fig.suptitle('Evolución de métricas')
    axs[0].plot(t, ise_v, label="ISE")
    axs[0].plot(t, iae_v, label="IAE")
    axs[1].plot(t, itse_v, label="ITSE")
    axs[1].plot(t, itae_v, label="ITAE")
    axs[1].set(xlabel='Tiempo (segs)')
    for ax in axs:
        ax.set(ylabel="Magnitud")
        ax.legend()
    plt.show()


def grafico_torques(t: np.array, torques_vector: np.array):
    plt.plot(t, torques_vector[:, 0], label="tau 1")
    plt.plot(t, torques_vector[:, 1], label="tau 2")
    plt.xlabel("Tiempo (segs)")
    plt.ylabel("Torques (N*m)")
    plt.title("Torques en el tiempo")
    plt.legend()
    plt.show()


def grafico_estados(t: np.array, x_vector: np.array, x_ref: np.array):
    x_v = x_vector[:, 0]
    y_v = x_vector[:, 1]
    x_ref_v = x_ref[:, 0]
    y_ref_v = x_ref[:, 1]
    fig, axs = plt.subplots(2)
    fig.suptitle('Tendencia a la referencia')
    axs[0].plot(t, x_v, label="Posicion x")
    axs[0].plot(t, x_ref_v, label="Referencia")
    axs[1].plot(t, y_v, label="Posicion y")
    axs[1].plot(t, y_ref_v, label="Referencia")
    axs[1].set(xlabel='Tiempo (segs)')
    for ax in axs:
        ax.set(ylabel='Distancia (mts)')
        ax.legend()
    plt.show()


def graficos_estados_todos(t, x_ref: np.array, x_v_pid_: np.array,
                           x_v_liq_: np.array, x_v_osc_: np.array):
    x_v_pid = x_v_pid_[:, 0]
    y_v_pid = x_v_pid_[:, 1]
    x_v_liq = x_v_liq_[:, 0]
    y_v_liq = x_v_liq_[:, 1]
    x_v_osc = x_v_osc_[:, 0]
    y_v_osc = x_v_osc_[:, 1]
    x_ref_v = x_ref[:, 0]
    y_ref_v = x_ref[:, 1]
    fig, axs = plt.subplots(2)
    fig.suptitle('Tendencia a la referencia')

    axs[0].plot(t, x_v_pid, label="pid")
    axs[0].plot(t, x_v_liq, label="liq")
    axs[0].plot(t, x_v_osc, label="osc")
    axs[0].plot(t, x_ref_v, label="Referencia")

    axs[1].plot(t, y_v_pid, label="pid")
    axs[1].plot(t, y_v_liq, label="liq")
    axs[1].plot(t, y_v_osc, label="osc")
    axs[1].plot(t, y_ref_v, label="Referencia")
    axs[1].set(xlabel='Tiempo (segs)')
    for ax in axs:
        ax.set(ylabel='Distancia (mts)')
        ax.legend()
    plt.show()


if __name__ == '__main__':
    t = np.linspace(0, 10, 3)
    ex = np.array([0.01, 0.11, -0.01])
    ey = np.array([0.01, 0.11, -0.01])*2

    grafico_estados(t, ex, ey)
