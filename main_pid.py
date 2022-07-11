from lib.util import control_de_referencia, gen_tiempo, dif_err_pid, theta_real
from lib.pid import pid_calc
import lib.graficos as graphs
import lib.indices as ind
import numpy as np
import glfw
import gym

# Quieres graficos? TODO
PLOTING = False
# Quieres indices?
INDICES = False


def main():
    # PARAMS
    Ts = 0.01
    test_steps = 5000
    if INDICES and PLOTING:
        test_steps = 1000
    # PID
    pid_k1 = np.array([0.1, .0, 0.1])
    pid_k2 = np.array([0.1, .0, 0.1])
    err_v1 = np.array([0.0, .0, 0.0])
    err_v2 = np.array([0.0, .0, 0.0])
    # torques inicial
    torques = np.array([0.0000, 0.000])
    # Genera vector de tiempo
    t = gen_tiempo(Ts, test_steps)
    # Para visualizar que es lo que hace el agente
    glfw.init()
    env = gym.make("Reacher-v2")  # brazo 2 DOF

    # Vectores para grafico TODO
    e_torques_t = np.zeros((len(t), 2))
    e_x_t = np.zeros((len(t), 2))
    theta_ref_t = np.zeros((len(t), 2))
    theta_real_t = np.zeros((len(t), 2))
    # Vectores de indice
    ise_t = np.zeros((len(t), 1))
    itse_t = np.zeros((len(t), 1))
    iae_t = np.zeros((len(t), 1))
    itae_t = np.zeros((len(t), 1))

    if INDICES and PLOTING:
        observation, _ = env.reset(seed=None, return_info=True)
        theta_ref = control_de_referencia(observation)

    # Iter
    for k in range(test_steps):
        # Cambiar seed
        if k % (test_steps/20) == 0 and not INDICES:
            observation, _ = env.reset(seed=None, return_info=True)
            theta_ref = control_de_referencia(observation)
        # Itera la interfaz
        env.render()
        accion = env.action_space.sample()
        # Aplica torque
        observation, reward, _, _ = env.step(torques)
        # Calculamos diferencia y lo guardamos en el vector buffer
        err_v1, err_v2, err_theta = dif_err_pid(
            observation, theta_ref, err_v1, err_v2)
        # Aplicamos ley de control PID
        torques = pid_calc(pid_k1, err_v1, pid_k2, err_v2, torques)
        if PLOTING:
            # GRAFICOS
            e_torques_t[k, :] = torques
            e_x_t[k, :] = err_theta
            theta_ref_t[k, :] = theta_ref
            theta_real_t[k, :] = theta_real(observation)
            if INDICES:
                ise_t[k] = ind.ise(e_x_t)
                itse_t[k] = ind.itse(e_x_t)
                iae_t[k] = ind.iae(e_x_t)
                itae_t[k] = ind.itae(e_x_t)

    glfw.terminate()
    env.close()

    if PLOTING:
        # Generar graficos TODO
        graphs.grafico_convergencia(t, e_x_t)
        graphs.grafico_torques(t, e_torques_t)
        graphs.grafico_estados(t, theta_real_t, theta_ref_t)
        if INDICES:
            graphs.grafico_indices(t, ise_t, itse_t, iae_t, itae_t)


if __name__ == '__main__':
    main()
