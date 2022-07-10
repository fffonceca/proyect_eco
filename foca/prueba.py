import numpy as np
import gym  # OpenAI gym
import glfw


def pid_calculation(pid_wights: np.array,
                    u_k_1: float, e_k_v: np.array, Ts=0.01) -> float:
    """
    Calculate new actuator
    args:
        pid_wights = [Kp, Ki, Kd]
        u_k_1: actuador anterior
        e_k = [e_k, e_k_1, e_k_2]

    return new actuator value
        u_new -> float
    """
    # despackaging e_k
    e_k, e_k_1, e_k_2 = e_k_v

    # despackaging PID
    Kp, Ki, Kd = pid_wights

    u_k = u_k_1 + Kp*(e_k-e_k_1) + Ki*e_k*Ts + Kd*(e_k-2*e_k_1+e_k_2)/Ts

    return u_k


def main():
    # Torque 1 params
    u_prev1 = 0
    error_vector1 = np.array([0.0, 0.0, 0.0])
    pid_constants1 = np.array([1.0, 0.2, 0.3])
    # Torque 2 params
    u_prev2 = 0
    error_vector2 = np.array([0.0, 0.0, 0.0])
    pid_constants2 = np.array([1.0, 0.2, 0.3])


    # Para visualizar que es lo que hace el agente
    glfw.init()

    test_steps = 1000

    env = gym.make("Reacher-v2")  # brazo 2 DOF
    observation, info = env.reset(seed=0, return_info=True)
    print(observation)

    for _ in range(test_steps):
        env.render()
        accion = env.action_space.sample()

        observation, reward, done, aux_dict = env.step([u_prev1, u_prev2])

        # # PID calc of torque 1
        # u_prev1 = pid_calculation(pid_constants1, u_prev1, error_vector1)
        # # PID calc of torque 2
        # u_prev2 = pid_calculation(pid_constants2, u_prev2, error_vector2)

        print(len(observation))

    glfw.terminate()
    env.close()


if __name__ == '__main__':
    main()
