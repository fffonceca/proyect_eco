from scipy.linalg import expm
import numpy as np


def matrices_completas(obs: np.array, torques: np.array):
    """
    Recibe el vector de observaciones, los torques y se obtienen
    las matrices completras para el LQI
    """
    # Matrices continuas
    A, B, C, D = jacobianos(obs, torques)
    # Matrices discretas
    Ad, Bd, Cd = discretizar_jacobianos(A, B, C)
    # Matrices aumentadas
    Aa, Ba, Ca = ampliar_matrices_LQI(Ad, Bd, Cd, D)

    return Aa, Ba, Ca


def jacobianos(observation: np.array, u: np.array,
               m=0.1*0.01**2*np.pi*1000, L1=0.1, L2=0.1):
    """
    Entrega las matrices en el espacio continuo
    """
    # Depackaging obs
    obs0, obs1, obs2, obs3, _, _, obs6, obs7, _, _, _,  = observation

    # Matriz A
    A = np.array([])

    # Matriz B
    den_b = L1**2*m*(16*L1**2+12*L1*L2*obs1+3*L2**2*(17-12*obs1**2))
    B = np.array([
        [0., 0.],
        [0., 0.],
        [12*(L1**2+3*L2**2), -36*L2*(L2+2*L1*obs1)],
        [-36*L2*(L2+2*L1*obs1), 36*L2**2+48*L1*(4*L1+3*L2*obs1)]
    ])/den_b

    # Matriz C
    C = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.]
    ])

    # Matriz D
    D = np.array([
        [0., 0.],
        [0., 0.]
    ])

    return A, B, C, D


def discretizar_jacobianos(A: np.array, B: np.array, C: np.array, Ts=0.01):
    """
    Discretiza las matrices
    Continuo -> Discreto
    """
    Ad = expm(A*Ts)
    Bd = ((Ad*Ts).dot(np.eye(len(Ad))-A*Ts/2.)).dot(B)
    return Ad, Bd, C


def ampliar_matrices_LQI(A, B, C, D):
    """
    Entrega a partir de las matrices A, B, C y D
    las matrices aumentadas para el control LQI
    """
    A = np.vstack((A, -C))
    n_filas = int(A.shape[0])
    a = np.zeros((n_filas, 2))
    Aa = np.hstack((A, a))
    Ba = np.vstack((B, -D))
    c = np.zeros((2, 2))
    Ca = np.hstack((C, c))

    return Aa, Ba, Ca
