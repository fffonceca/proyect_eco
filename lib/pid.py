import numpy as np


def pid_calc(pid_wights1: np.array, e_k_v_1: np.array,
             pid_wights2: np.array, e_k_v_2: np.array,
             torques: np.array, Ts=0.01) -> tuple:
    """
    Calculate new actuators value
    args:
        pid_wights = [Kp, Ki, Kd]
        u_k_1: actuador anterior
        e_k = [e_k, e_k_1, e_k_2]

    return new actuator value
        (torque_1, torque_2) -> (float, float)
    """

    # depackaging torques
    u_prev1, u_prev2 = torques
    # Torque 2
    # depackaging e_k
    e_k, e_k_1, e_k_2 = e_k_v_1
    # despackaging PID
    Kp, Ki, Kd = pid_wights1
    u_new1 = u_prev1 + Kp*(e_k-e_k_1) + Ki*e_k*Ts + Kd*(e_k-2*e_k_1+e_k_2)/Ts

    # Torque 2
    # depackaging e_k
    e_k, e_k_1, e_k_2 = e_k_v_2
    # despackaging PID
    Kp, Ki, Kd = pid_wights2
    u_new2 = u_prev2 + Kp*(e_k-e_k_1) + Ki*e_k*Ts + Kd*(e_k-2*e_k_1+e_k_2)/Ts

    return np.array([u_new1, u_new2])
