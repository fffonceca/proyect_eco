import numpy as np
import gym  # OpenAI gym
import glfw
from funciones_varias import *

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
        (u_new1, u_new2) -> (float, float)
    """
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


def main():
    # PARAMS
    Ts = 0.01
    # PID
    pid_k1 = np.array([1.0, 0.2, 0.3])
    pid_k2 = np.array([1.0, 0.2, 0.3])
    # prev
    torques = np.array([0.0, 0.0])
    # Errors
    err_v1 = np.array([0.0, 0.0, 0.0])
    err_v2 = np.array([0.0, 0.0, 0.0])

    # Para visualizar que es lo que hace el agente
    glfw.init()

    test_steps = 1000

    env = gym.make("Reacher-v2")  # brazo 2 DOF
    observation, info = env.reset(seed=0, return_info=True)
    print(observation)

    for _ in range(test_steps):
        env.render()
        accion = env.action_space.sample()

        observation, reward, done, aux_dict = env.step(torques)
        theta1 = angulo_real(observation[0], observation[2])
        theta2 = angulo_real(observation[1], observation[3])
        x_ref,y_ref = observation[4], observation[5]
        theta_refR, theta_refL, posible = theta_ref12({'L1': 0.1, 'L2': 0.1},
                                            x_ref, y_ref)
        theta_ref = control_referencia(theta_refR, theta_refL, theta1)

        error_theta1 = theta_ref[0] - theta1
        error_theta2 = theta_ref[1] - theta2

        
        # PID calc
        torques = pid_calc(pid_k1, err_v1, pid_k2, err_v2, torques, Ts)

        print(observation[0], observation[1])

    glfw.terminate()
    env.close()


if __name__ == '__main__':
    main()