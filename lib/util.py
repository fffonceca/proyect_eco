import numpy as np


def gen_tiempo(Ts: float, steps: int) -> np.array:
    """
    Genera el vector de tiempo
    """
    return np.linspace(0, Ts*steps-1, Ts*steps)


def output_info(observation: np.array, torques) -> tuple:
    """
    Calcula salidas para grafico
    """
    err_eucl = np.sqrt(observation[8]**2 + observation[9]**2)
    tor_norm = np.sqrt(torques[0]**2 + torques[1]**2)
    err_x_y = np.array(observation[4:6])

    return (err_eucl, err_x_y, tor_norm)
