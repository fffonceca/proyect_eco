from lib.util import control_de_referencia, gen_tiempo, dif_err_pid
from lib.osc import control_osc
import numpy as np
import glfw
import gym


def main():
    # PARAMS
    Ts = 0.01
    test_steps = 10000
    # torques inicial
    torques = np.array([0.0, 0.0])
    # Genera vector de tiempo
    t = gen_tiempo(Ts, test_steps)
    # Para visualizar que es lo que hace el agente
    glfw.init()
    env = gym.make("Reacher-v2")  # brazo 2 DOF
    observation, _ = env.reset(seed=None, return_info=True)

    for k in range(test_steps):
        # Cambiar seed
        if k % (test_steps/20) == 0:
            observation, _ = env.reset(seed=None, return_info=True)
            torques = np.array([0.0, 0.0])
        # Itera la interfaz
        env.render()
        accion = env.action_space.sample()
        # Aplica torque
        observation, reward, _, _ = env.step(torques)
        # Aplicamos ley de control PID
        torques = control_osc(observation, kp=5, kv=5)

    glfw.terminate()
    env.close()


if __name__ == '__main__':
    main()
