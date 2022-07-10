import numpy as np
import gym
import glfw


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
