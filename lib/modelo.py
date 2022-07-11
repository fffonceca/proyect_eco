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


def jacobianos(observation: np.array, tau: np.array,
               m=0.1*0.01**2*np.pi*1000, L1=0.1, L2=0.1):
    """
    Entrega las matrices en el espacio continuo

    obs0: Cos[q1[t]]
    obs1: Cos[q2[t]]
    obs2: Sin[q1[t]]
    obs3: Sin[q2[t]]
    obs6: omega_1
    obs7: omega_2
    """
    # Depackaging obs
    obs0, obs1, obs2, obs3, _, _, obs6, obs7, _, _, _ = observation

    COS2Q2 = obs1**2 - obs3**2

    A32 = (6*L2*(-2*(L1-6*L2*obs1)*obs3*(
        -6*(
            L1**2+3*L2**2
        )*m*(
            L1*L2*m*obs3*obs7*(2*obs6+obs7)+2*tau[0]
        )-18*L2*m*(L2+2*L1*obs1)*(L1*L2*m*obs3*obs6**2-2*tau[1])
        )-L1*m*(16*L1**2+12*L1*L2*obs1+3*L2**2*(17-12*obs1**2))*(
            -m*(6*L1*L2*(obs1**2-obs3**2)*obs6**2+3*L2**2*obs1*(
                obs6+obs7
            )**2+L1**2*obs1*obs7*(2*obs6+obs7))-12*obs3*tau[1]))
    )/(L1**2*m**2*(16*L1**2+12*L1*L2*obs1+3*L2**2*(17-12*obs1**2))**2)

    A33 = -1*(12*L2*obs3*(6*L1*L2*obs1*obs6+L1**2*obs7+3*L2**2*(obs6+obs7)))/(L1*(16*L1**2+33*L2**2+12*L1*L2*obs1-18*L2**2*COS2Q2))

    A34 = -1*(36*L2**2*(L2*+2*L1*obs1)*obs3*(obs6+obs7)) / (L1*(16*L1**2 + 33*L2**2 + 6*L2*(2*L1*obs1 - 3*L2*COS2Q2)))

    A42 = (6*L2*(
        L1*m*(-256*L1**4*obs1*obs6**2+9*L2**4*obs1*(
            -17+12*obs1**2+24*obs3**2
        )*(obs6+obs7)**2+24*L1**2*L2**2*obs1*(
            6*(-6+3*obs1**2+8*obs3**2)*obs6**2-2*(
                2+3*obs1**2
            )*obs6*obs7-(2+3*obs1**2)*obs7**2
        )-96*L1**3*L2*(
            -obs3**2*obs7*(2*obs6+obs7)+obs1**2*(
                4*obs6**2+2*obs6*obs7+obs7**2
            )
        )+18*L1*L2**3*(
            12*obs1**4*(
                2*obs6**2+2*obs6*obs7+obs7**2
            )+obs3**2*(
                32*obs6**2+30*obs6*obs7+15*obs7**2
            )+obs1**2*(12*(-3+2*obs3**2)*obs6**2+2*(
                -19+12*obs3**2
            )*obs6*obs7+(-19+12*obs3**2)*obs7**2)))+12*(
                16*L1**3+36*L2**3*obs1+9*L1*L2**2*(5+4*obs1**2)
            )*obs3*tau[0]-144*L2*(L2+2*L1*obs1)*(8*L1+3*L2*obs1)*obs3*tau[1])
        )/(L1**2*m*(16*L1**2+12*L1*L2*obs1+3*L2**2*(17-12*obs1**2))**2)

    A43 = -1*(12*L2*obs3*(16*L1**2*obs6+3*L2**2*(obs6+obs7) + 6*L1*L2*obs1*(2*obs6+obs7)))/(L1*(16*L1**2 + 12*L1*L2*obs1 + 3*L2**2*(17-12*obs1**2)))

    A44 = -1*(36*L1**2*(L2 + 2*L1*obs1)*obs3*(obs6 + obs7)) / (L1*(16*L1**2+12*L1*L2*obs1 + 3*L2**2*(17-12*obs1**2)))

    # Matriz A
    A = np.array([
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [0., A32, A33, A34],
        [0., A42, A43, A44]
    ])

    # Matriz B
    den_b = L1**2*m*(16*L1**2+12*L1*L2*obs1+3*L2**2*(17-12*obs1**2))
    B = np.array([
        [0., 0.],
        [0., 0.],
        [12*(L1**2+3*L2**2), -36*L2*(L2+2*L1*obs1)],
        [-36*L2*(L2+2*L1*obs1), 36*L2**2+48*L1*(4*L1+3*L2*obs1)]
    ])/den_b

    # Matriz C
    C = np.eye(4)

    # Matriz D
    D = np.array([
        [0., 0.],
        [0., 0.],
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
    a = np.zeros((n_filas, 4))
    Aa = np.hstack((A, a))
    Ba = np.vstack((B, -D))
    c = np.zeros((4, 4))
    Ca = np.hstack((C, c))

    return Aa, Ba, Ca
