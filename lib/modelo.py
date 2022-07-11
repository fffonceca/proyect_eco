import numpy as np


def jacobiano(observation: np.array, u: np.array,
              m=0.1*0.01**2*np.pi*1000, L1=0.1, L2=0.1):
    obs0, obs1, obs2, obs3, _, _, obs6, obs7, _, _, _,  = observation
    A = np.array([])

    den_b = L1**2*m*(16*L1**2+12*L1*L2*obs1+3*L2**2*(17-12*obs1**2))
    B = np.array([
        [0., 0.],
        [0., 0.],
        [12*(L1**2+3*L2**2), -36*L2*(L2+2*L1*obs1)],
        [-36*L2*(L2+2*L1*obs1), 36*L2**2+48*L1*(4*L1+3*L2*obs1)]
    ])/den_b

    C = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.]
    ])

    D = np.array([
        [0., 0.],
        [0., 0.]
    ])


def matrices_aumentadas(A, B, C, D):
    # Matriz Aa
    aux = np.vstack((A, -C))
    n_filas = aux.shape[0]
    a = np.zeros((n_filas, 3))
    Aa = np.hstack((aux, a))

    # Matriz Ba
    Ba = np.vstack((B, -D))

    # Matriz Ca
    Ca = 0
    Ca = np.hstack((C, np.zeros((3, 3))))

    Da = D

    return Aa, Ba, Ca, Da
