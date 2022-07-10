import numpy as np
from scipy.linalg import expm
from scipy.linalg import solve_discrete_are
"""
Para el control LQR-LQI es necesario usar las ecuaciones dinámicas
obtenidas a partir de Lagrange, donde se obtiene una ecuación para tau_1
y tau_2, estas quedan en función de thetas y sus derivadas las cuales 
pueden ser extraídas directamente del vector de estados 
"""

def extender_estado(x, error_theta1, error_theta2):
    """
    el estado se extiende de manera vertical OwO
    """
    
    x = np.vstack((x,error_theta1))
    x = np.vstack((x,error_theta2))
    return x


def torques(obs: np.array, obs_x_1: np.array,  L1=0.1, L2=0.1, m1=0.1, m2=0.1, Ts = 0.01, g = 9.81): 
    # Se necesita el vector de estado anterior y el actual.
    
    # tau1, tau2 = M* theta'' + c + g
    # (theta'') = theta''1, theta''2
    theta_dot2 = np.array([ [(obs[6]-obs_x_1[6])/Ts],
                        [(obs[7]-obs_x_1[7])/Ts]])

    M = np.array([ [  m1*L1**2 + m2*(L1**2 + 2*L1*L2*obs[1] + L2**2)  ,  m2*(L1*L2*obs[1] + L2**2)] , 
             [m2*(L1*L2*obs[1] + L2**2) ,     m2 * L2**2 ]])

    c = np.array([-m2*L1*L2+ obs[3] * (2*obs[6]*obs[7] + (obs[7])**2)],
               [m2*L1*L2*(obs[6]**2)*obs[3]])
            
    g = np.array([[(m1 + m2) * L1*g*obs[0] +  m2*g*L2*(obs[0]*obs[1] - obs[2]*obs[3])],
             [m2*g*L2*(obs[0]*obs[1] - obs[2]*obs[3])]   ])
    
    tau = M*theta_dot2 + c + g

    return tau

def SmallestSignedAngleBetween(theta1, theta2):
    a = (theta1 - theta2) % (2.*np.pi)
    b = (theta2 - theta1) % (2.*np.pi)
    return -a if a < b else b


def ampliar_matrices_LQI(A,B,C,D):
    A = np.vstack((A, -C))
    n_filas = int(A.shape[0])
    a = np.zeros((n_filas, 2))
    A = np.hstack((A,a))
    B = np.vstack((B, -D))
    c = np.zeros((2, 2))
    C = np.hstack((C, c))
    return A, B, C

def discretizar_jacobianos(A,B, Ts = 0.01):
    Ad = expm(A*Ts) 
    Bd = ((Ad*Ts).dot(np.eye(len(Ad))-A*Ts/2.)).dot(B)
    return Ad, Bd

def gananciaK(A, B, Q_lqr, R_lqr):
    P = solve_discrete_are(A, B, Q_lqr, R_lqr)
    K = np.linalg.inv(R_lqr+B.T.dot(P).dot(B)).dot(B.T).dot(P).dot(A)
    return K 

def torque_lqr(x, u, K, instante_t, w_lqr=2.):
    u_lqr = -K.dot(x[instante_t,:]-x[instante_t - 1,:])
    u[instante_t,:] = w_lqr*u_lqr


