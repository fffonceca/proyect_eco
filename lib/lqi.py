from scipy.linalg import solve_discrete_are
from lib.util import theta_real
import numpy as np
"""
Para el control LQR-LQI es necesario usar las ecuaciones dinámicas
obtenidas a partir de Lagrange, donde se obtiene una ecuación para tau_1
y tau_2, estas quedan en función de thetas y sus derivadas las cuales
pueden ser extraídas directamente del vector de estados
"""

Q_a = np.diag([2., 2., 1., 1., .1, .1, .1, .1])
R_a = np.diag([1., 1.])


def gananciaK(Aa: np.array, Ba: np.array, Qa: np.array, Ra: np.array):
    """
    Recibe las matrices aumentadas y obtiene
    la ganancia K a partir de las matrices
    Aa, Ba, Qa, Ra
    """
    P = solve_discrete_are(Aa, Ba, Qa, Ra)
    K = np.linalg.inv(Ra+Ba.T.dot(P).dot(Ba)).dot(Ba.T).dot(P).dot(Aa)
    return K


def vector_estados(obs: np.array) -> np.array:
    """
    Se obtiene el vector de estados
    necesario para el LQI

    """
    theta1, theta2 = theta_real(obs)

    return np.array([theta1, theta2, obs[6], obs[7]])


if __name__ == '__main__':
    pass
