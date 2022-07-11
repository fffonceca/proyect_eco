from lib.util import control_de_referencia, gen_tiempo, dif_err_pid
from lib.pid import pid_calc
import numpy as np
import glfw
import gym


def main():
    # PARAMS
    Ts = 0.01
    test_steps = 10000
    # PID
    pid_k1 = np.array([0.1, 0., 0.1])
    pid_k2 = np.array([0.1, 0., 0.1])
    err_v1 = np.array([0.0, 0.0, 0.0])
    err_v2 = np.array([0.0, 0.0, 0.0])
    # torques inicial
    torques = np.array([0.0000, 0.000])

    # Genera vector de tiempo
    t = gen_tiempo(Ts, test_steps)

    # Para visualizar que es lo que hace el agente
    glfw.init()
    env = gym.make("Reacher-v2")  # brazo 2 DOF

    for k in range(test_steps):
        # Cambiar seed
        if k % (test_steps/20) == 0:
            observation, _ = env.reset(seed=None, return_info=True)
            theta_ref = control_de_referencia(observation)
        # Itera la interfaz
        env.render()
        accion = env.action_space.sample()
        # Aplica torque
        observation, reward, _, _ = env.step(torques)
        # Calculamos diferencia y lo guardamos en el vector buffer
        err_v1, err_v2 = dif_err_pid(observation, theta_ref, err_v1, err_v2)
        # Aplicamos ley de control PID
        torques = pid_calc(pid_k1, err_v1, pid_k2, err_v2, torques)

    glfw.terminate()
    env.close()


if __name__ == '__main__':
    main()
