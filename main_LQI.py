from lib.util import control_de_referencia, gen_tiempo, dif_err_lqi
from lib.modelo import matrices_completas
from lib.lqi import vector_estados, gananciaK, Q_a, R_a
import numpy as np
import glfw
import gym

# Quieres graficos?
PLOTING = False
# Quieres indices?
INDICES = False


def main():
    # PARAMS
    Ts = 0.01
    test_steps = 10000
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
        if k % (test_steps/20) == 0:
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

        # Itera la interfaz
        env.render()
        accion = env.action_space.sample()
        # Aplica torque
        observation, reward, _, _ = env.step(torques)

    glfw.terminate()
    env.close()


if __name__ == '__main__':
    main()
