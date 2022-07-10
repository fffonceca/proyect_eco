import numpy as np


def control_de_referencia(obs: np.array) -> np.array:
    """
    Devuelve el objetivo de referencia en funcion del estado y de la posicion
    del target.

    retorna un arreglo con [theta1_ref, theta2_ref]
    """
    sol1, sol2, val = theta_ref12(obs)
    return ref_control(sol1, sol2, obs[0])


def dif_err(obs: np.array, theta_r: np.array, e_v1: np.array, e_v2: np.array):
    """
    Calcula la diferencia entre la referencia y el
    valor actual del angulo. Lo guarda en los vectores
    y los retorna.
    """
    theta = theta_real(obs)
    err_theta = theta_r - theta
    # print(f"Theta1: {theta[0]*180/np.pi}, Theta ref: {theta_r[0]*180/np.pi}")
    e_v1 = np.hstack((err_theta[0], e_v1[0:2]))
    e_v2 = np.hstack((err_theta[1], e_v2[0:2]))

    return e_v1, e_v2


def gen_tiempo(Ts: float, steps: int) -> np.array:
    """
    Genera el vector de tiempo
    """
    return np.linspace(0, round(Ts*steps-1), round(Ts*steps))


def distancia(pos1, pos2):
    """
    Calcula distancia relativa entre dos puntos
    """
    return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)


def posicion(ang):
    if 0 <= ang < np.pi/2:
        x = np.cos(ang)
        y = np.sin(ang)
    if np.pi/2 <= ang < np.pi:
        x = np.cos(ang)
        y = -np.sin(ang)
    if np.pi <= ang < 3*np.pi/2:
        x = -np.cos(ang)
        y = -np.sin(ang)
    else:
        x = -np.cos(ang)
        y = np.sin(ang)

    return np.array([x, y])


def theta_ref12(obs: np.array, L1=0.1, L2=0.1):
    """
    obs: vector de estados
    L1: largo del primer brazo
    L2: largo del segundo brazo

    retorna sol1, sol2, result
    Donde sol1 y sol2 son las soluciones righty y lefty
    respectivamente
    """
    x, y = target_pos(obs)
    # El brazo no es tan largo
    if np.sqrt(x**2 + y**2) > L1 + L2:
        return 0, 0, False

    # Menor a lo que puede el brazo
    if np.sqrt(x**2 + y**2) < L1 - L2:
        return 0, 0, False

    beta = np.arccos((L1**2+L2**2-(x**2+y**2))/(2*L1*L2))
    alpha = np.arccos((x**2+y**2+L1**2-L2**2)/(2*L1*np.sqrt(x**2+y**2)))
    gamma = np.arctan2(y, x)

    sol1 = np.array([gamma - alpha, np.pi - beta])
    sol2 = np.array([gamma + alpha, np.pi + beta])
    return sol1, sol2, True


def ref_control(thetaR: np.array, thetaL: np.array, theta: float) -> np.array:
    """
    Mediante minimizacion, calcula menor distancia
    desde el primer brazo hacia cualesquiera de las
    dos soluciones del target.

    De la distancia menor se obtiene el theta_ref
    respectivo
    """
    posR = posicion(thetaR[0])
    posL = posicion(thetaL[0])
    pos = posicion(theta)

    dR = distancia(posR, pos)
    dL = distancia(posL, pos)

    if dR > dL:
        theta_ref = thetaL
    else:
        theta_ref = thetaR

    return theta_ref


def target_pos(obs: np.array):
    """
    Obtiene pos del target en x e y
    """
    return np.array([obs[4], obs[5]])


def theta_real(obs: np.array) -> np.array:
    """
    Calcula a partir del vector de obs
    los valores del estado de los angulos
    de los brazos

    """
    theta1 = angulo_real(obs[0], obs[2])
    theta2 = angulo_real(obs[1], obs[3])
    return np.array([theta1, theta2])


def angulo_real(cos, sin):
    """
    Calcula angulos desde el eje referencial,
    cos = cos(theta) y sin = sin(theta)

    return theta
    """
    if sin > 0:
        theta = np.arccos(cos)
    else:
        theta = -1*np.arccos(cos)
    return theta


def output_info(observation: np.array, torques) -> tuple:
    """
    Calcula salidas para grafico
    """
    err_eucl = np.sqrt(observation[8]**2 + observation[9]**2)
    tor_norm = np.sqrt(torques[0]**2 + torques[1]**2)
    err_x_y = np.array(observation[4:6])

    return (err_eucl, err_x_y, tor_norm)


if __name__ == '__main__':
    pass
