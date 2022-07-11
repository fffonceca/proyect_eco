from lib.util import control_de_referencia, gen_tiempo, dif_err_lqi, theta_real
from lib.lqi import vector_estados, gananciaK, Q_a, R_a
from lib.modelo import matrices_completas
import lib.graficos as graphs
import lib.indices as ind
import numpy as np
import glfw
import gym

# Quieres graficos?
PLOTING = True
# Quieres indices?
INDICES = True
# Quieres ruido?
NOISE = True


def main():
    # PARAMS
    Ts = 0.01
    test_steps = 5000
    if INDICES and PLOTING:
        test_steps = 1000
    # torques inicial
    torques = np.array([0.0000, 0.000])
    # Genera vector de tiempo
    t = gen_tiempo(Ts, test_steps)
    # Integrador del error
    sum_err = np.array([0., 0., 0., 0.])

    # Para visualizar que es lo que hace el agente
    glfw.init()
    env = gym.make("Reacher-v2")  # brazo 2 DOF
    if INDICES and PLOTING:
        observation, _ = env.reset(seed=None, return_info=True)
        theta_ref = control_de_referencia(observation)

    # Vectores para grafico
    e_torques_t = np.zeros((len(t), 2))
    e_x_t = np.zeros((len(t), 2))
    theta_ref_t = np.zeros((len(t), 2))
    theta_real_t = np.zeros((len(t), 2))
    # Vectores de indice
    ise_t = np.zeros((len(t), 1))
    itse_t = np.zeros((len(t), 1))
    iae_t = np.zeros((len(t), 1))
    itae_t = np.zeros((len(t), 1))

    for k in range(test_steps):
        # Cambiar seed
        if k % (test_steps/20) == 0 and not INDICES:
            observation, _ = env.reset(seed=None, return_info=True)
            theta_ref = control_de_referencia(observation)
        # Obtener las matrices completas
        Aa, Ba, Ca = matrices_completas(observation, torques)
        # Calculo de error en "y"
        err_vector = dif_err_lqi(observation, theta_ref)
        sum_err = sum_err + err_vector*Ts
        """
            Control LQI
        """
        # Ganancia K
        K = gananciaK(Aa, Ba, Q_a, R_a)
        # Vector de estados
        theta1, theta2, omega1, omega2 = vector_estados(observation)
        # Generar vector aumentado 8x1
        x_a = np.transpose(
            np.hstack((err_vector[0:2], -omega1, -omega2, sum_err))
        )
        # Calculo de actuadores
        torques = K.dot(x_a)
        # Si hay ruido TODO, agregamos a la medicion
        if NOISE:
            torques += 0.001*np.random.normal(size=2)

        # Itera la interfaz
        env.render()
        accion = env.action_space.sample()
        # Aplica torque
        observation, reward, _, _ = env.step(torques)

        if PLOTING:
            # GRAFICOS
            e_torques_t[k, :] = torques
            e_x_t[k, :] = err_vector[0:2]
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
