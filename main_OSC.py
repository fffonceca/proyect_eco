from lib.util import control_de_referencia, gen_tiempo, dif_err_pid, theta_real
from lib.osc import control_osc
import lib.graficos as graphs
import lib.indices as ind
import numpy as np
import glfw
import gym

# Quieres graficos? TODO
PLOTING = True
# Quieres indices?
INDICES = True
# Quieres ruido?
NOISE = False
# Exportar?
EXPORT = True


def main():
    # PARAMS
    Ts = 0.01
    test_steps = 5000
    if INDICES and PLOTING:  # TODO
        test_steps = 1000
    # torques inicial
    torques = np.array([0.0, 0.0])
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

    if INDICES and PLOTING:  # TODO
        observation, _ = env.reset(seed=0, return_info=True)
        theta_ref = control_de_referencia(observation)

    for k in range(test_steps):
        # Cambiar seed
        if k % (test_steps/20) == 0 and not INDICES:  # TODO
            observation, _ = env.reset(seed=None, return_info=True)
            theta_ref = control_de_referencia(observation)
        # Itera la interfaz
        env.render()
        accion = env.action_space.sample()
        # Aplica torque
        observation, reward, _, _ = env.step(torques)
        # Aplicamos ley de control PID
        torques = control_osc(observation, kp=5, kv=5)
        # Si hay ruido TODO, agregamos a la medicion
        if NOISE:
            torques += 0.001*np.random.normal(size=2)

        # TODO
        if PLOTING:
            err_theta = dif_err_pid(observation, theta_ref, None, None)
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

    # TODO
    if PLOTING:
        # Generar graficos
        graphs.grafico_convergencia(t, e_x_t)
        graphs.grafico_torques(t, e_torques_t)
        graphs.grafico_estados(t, theta_real_t, theta_ref_t)
        if INDICES:
            graphs.grafico_indices(t, ise_t, itse_t, iae_t, itae_t)

    if EXPORT:
        np.save(graphs.OSC_PATH_EXPORT+"_state.npy",
                np.hstack((theta_real_t, theta_ref_t)))


if __name__ == '__main__':
    main()
