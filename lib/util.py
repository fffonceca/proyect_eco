import numpy as np


def control_de_referencia(obs: np.array) -> np.array:
    """
    Devuelve el objetivo de referencia en funcion del estado y de la posicion
    del target.

    retorna un arreglo con [theta1_ref, theta2_ref]
    """
    sol1, sol2, val = theta_ref12(obs)
    return ref_control(sol1, sol2, obs[0])


def dif_err_pid(obs: np.array, theta_r: np.array, e_v1: np.array,
                e_v2: np.array) -> np.array:
    """
    Calcula la diferencia entre la referencia y el
    valor actual del angulo. Lo guarda en los vectores
    y los retorna. Para PID
    """
    # Depackaging theta's
    theta1, theta2 = theta_real(obs)
    theta_ref1, theta_ref2 = theta_r

    # Calculates smallest signed angle
    # err_theta = theta_r - theta
    err_theta = np.array([
        smallest_angle(theta1, theta_ref1),
        smallest_angle(theta2, theta_ref2)
    ])

    e_v1 = np.hstack((err_theta[0], e_v1[0:2]))
    e_v2 = np.hstack((err_theta[1], e_v2[0:2]))

    return e_v1, e_v2, err_theta


def dif_err_lqi(obs: np.array, theta_r: np.array) -> np.array:
    # Depackaging theta's
    theta1, theta2 = theta_real(obs)
    theta_ref1, theta_ref2 = theta_r

    # Calculates smallest signed angle
    err_theta = np.array([
        smallest_angle(theta1, theta_ref1),
        smallest_angle(theta2, theta_ref2)
    ])

    return np.hstack((err_theta, -obs[6], -obs[7]))


def gen_tiempo(Ts: float, steps: int) -> np.array:
    """
    Genera el vector de tiempo
    """
    return np.linspace(0, round(Ts*steps-1), steps)


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


def vel_real(obs: np.array, L1=0.1, L2=0.1) -> np.array:
    """
    Calcula las velocidades vx vy
    del footprint

    obs0: Cos[q1[t]]
    obs1: Cos[q2[t]]
    obs2: Sin[q1[t]]
    obs3: Sin[q2[t]]
    obs6: q1'[t]
    obs7: q2'[t]
    """
    theta1, theta2 = theta_real(obs)

    vx = -L1*obs[2]*obs[6]-1/2*L2*np.sin(theta1+theta2)*(obs[6]+obs[7])
    vy = L1*obs[0]*obs[6]+1/2*L2*np.cos(theta1+theta2)*(obs[6]+obs[7])

    return np.array([vx, vy])


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
        print("No se puede, max")
        return 0, 0, False

    # Menor a lo que puede el brazo
    if np.sqrt(x**2 + y**2) < L1 - L2:
        print("No se puede, min")
        return 0, 0, False

    beta = np.arccos((L1**2+L2**2-x**2-y**2)/(2*L1*L2))
    alpha = np.arccos((x**2+y**2+L1**2-L2**2)/(2*L1*np.sqrt(x**2+y**2)))
    gamma = np.arctan2(y, x)

    sol1 = np.array([gamma - alpha, np.pi - beta])
    sol2 = np.array([gamma + alpha, beta - np.pi])
    # print(f"Sol1 {sol1*180/np.pi}, Sol2 {sol2*180/np.pi}")
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


def smallest_angle(theta_state, theta_ref):
    a = (theta_state - theta_ref) % (2.*np.pi)
    b = (theta_ref - theta_state) % (2.*np.pi)
    if a < b:
        return -a
    return b


if __name__ == '__main__':
    ref = 180
    estado = -175
    a = smallest_angle(estado*np.pi/180, ref*np.pi/180)
    print(a*180/np.pi)
