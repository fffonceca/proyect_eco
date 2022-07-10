import numpy as np
import gym
import glfw
import os
import time


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
    torques = np.array([0.001, 0.00])
    # Errors
    err_v1 = np.array([0.0, 0.0, 0.0])
    err_v2 = np.array([0.0, 0.0, 0.0])

    # Para visualizar que es lo que hace el agente
    glfw.init()
    test_steps = 100000

    env = gym.make("Reacher-v2")  # brazo 2 DOF
    observation, info = env.reset(seed=0, return_info=True)

    for _ in range(test_steps):
        env.render()
        accion = env.action_space.sample()

        observation, reward, done, aux_dict = env.step(torques)

        aux = np.sqrt(observation[8]**2 + observation[9]**2)
        # print(f"Distancia a target {aux}, pos target: {observation[4:6]}")
        print(accion)

    glfw.terminate()
    env.close()


if __name__ == '__main__':
    main()
