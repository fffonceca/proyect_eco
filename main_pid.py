from lib.util import control_de_referencia, gen_tiempo, dif_err
from lib.pid import pid_calc
import numpy as np
import glfw
import gym


def main():
    # PARAMS
    Ts = 0.01
    test_steps = 100000
    # PID
    pid_k1 = np.array([0.1, 0., 0.1])
    pid_k2 = np.array([0.1, 0., 0.1])
    err_v1 = np.array([0.0, 0.0, 0.0])
    err_v2 = np.array([0.0, 0.0, 0.0])
    # torques inicial
    torques = np.array([0.0, 0.0])

    # Genera vector de tiempo
    t = gen_tiempo(Ts, test_steps)

    # Para visualizar que es lo que hace el agente
    glfw.init()
    env = gym.make("Reacher-v2")  # brazo 2 DOF
    observation, _ = env.reset(seed=0, return_info=True)

    # theta_ref = control_de_referencia(observation)
    theta_ref = np.array([np.pi/2, np.pi/4])

    err_v1, err_v2 = dif_err(observation, theta_ref, err_v1, err_v2)

    torques = pid_calc(pid_k1, err_v1, pid_k2, err_v2, torques)

    for _ in range(test_steps):
        # Itera la interfaz
        env.render()
        # ?
        accion = env.action_space.sample()
        # Aplica torque
        observation, reward, _, _ = env.step(torques)
        # Calculamos diferencia y lo guardamos en el vector buffer
        err_v1, err_v2 = dif_err(observation, theta_ref, err_v1, err_v2)
        # print(err_v1)
        # Aplicamos ley de control PID
        torques = pid_calc(pid_k1, err_v1, pid_k2, err_v2, torques)
        # print(f"torques: {torques}")

    glfw.terminate()
    env.close()


if __name__ == '__main__':
    main()
