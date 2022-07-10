import numpy as np

"""
Para el control LQR-LQI es necesario usar las ecuaciones dinámicas
obtenidas a partir de Lagrange, donde se obtiene una ecuación para tau_1
y tau_2, estas quedan en función de thetas y sus derivadas las cuales 
pueden ser extraídas directamente del vector de estados 
"""

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


