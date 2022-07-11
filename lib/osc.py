from lib.util import theta_real, vel_real
from numpy.linalg import inv
from lib.modelo import MASA
import numpy as np


def mass_matrix(obs: np.array, L1=.1, L2=.1) -> np.array:
    """
    Calcula la matriz de inercia

    """
    M = np.array([
        [
            MASA*L1**2+MASA*(L1**2+2*L1*L2*obs[1]+L2**2),
            MASA*(L1*L2*obs[1]+L2**2)],
        [
            MASA*(L1*L2*obs[1]+L2**2),
            MASA*L2**2]
    ])

    return M


def mass_matrix_ee(M: np.array, J_ee: np.array) -> np.array:
    """
    Mass matrix e effector

    """
    return inv(J_ee.dot(inv(M)).dot(np.transpose(J_ee)))


def cor_centri_vector(obs: np.array, L1=.1, L2=.1) -> np.array:
    """
    Calcula el vector con torques de coriolis y
    torque centripeta.

    """
    c = np.array(
        [-MASA*L1*L2*obs[3]*(2*obs[6]*obs[7]+obs[7]**2)],
        [MASA*L1*L2*obs[6]**2*obs[3]]
    )

    return c


def Jacobian_ee(obs: np.array, L1=.1, L2=.1) -> np.array:
    """
    Obtiene el jacobiano, recordar:

    obs0: Cos[q1[t]]
    obs1: Cos[q2[t]]
    obs2: Sin[q1[t]]
    obs3: Sin[q2[t]]
    obs6: omega_1
    obs7: omega_2
    """
    theta1, theta2 = theta_real(obs)

    J_ee = np.array([
        [-L1*obs[2]-L1*np.sin(theta1+theta2), -L1*np.sin(theta1+theta2)],
        [L1*obs[0]+L1*np.cos(theta1+theta2), L1*np.cos(theta1+theta2)]
    ])

    return J_ee


def control_osc(obs: np.array, kp=1, kv=0.5) -> np.array:
    """
    Calcula los torques para el control OSC
    torque: tau1, tau2

    obs[8]: x_fingerprint - x_target
    obs[9]: y_fingerprint - y_target

    https://studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
    """
    # Obtener vectores de error
    pos_error = -np.array(obs[8:10])
    vel_error = -vel_real(obs)

    J_ee = Jacobian_ee(obs)
    M = mass_matrix(obs)
    Mxee = mass_matrix_ee(M, J_ee)

    return np.transpose(J_ee).dot(Mxee).dot(kp*pos_error+kv*vel_error)
