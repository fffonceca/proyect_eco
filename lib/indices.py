import numpy as np


# Indices de desempeño
# Todos los indices se hicieron con
# aproximacion trapezoidal
def ise(error_vector: np.array, Ts=0.01) -> float:
    """
    Integral square-error criterion
    (criterio de la integral del error al cuadrado)

    """
    error_vector = err_vec2_to_err_vec(error_vector)
    sum = error_vector[0]**2
    for index in range(1, len(error_vector)):
        sum += error_vector[index]**2 + error_vector[index-1]**2

    return sum*Ts/2


def itse(error_vector: np.array, Ts=0.01) -> float:
    """
    Integral of time multiplied squared error criterion
    (Integral del error cuadrado multiplicado por el tiempo)

    """
    error_vector = err_vec2_to_err_vec(error_vector)
    sum = 0
    for index in range(1, len(error_vector)):
        sum += index*(error_vector[index]**2 + error_vector[index-1]**2)

    return sum*Ts/2


def iae(error_vector: np.array, Ts=0.01) -> float:
    """
    Integral Absolute Error Criterion
    (Criterio de la integral del valor absoluto del error)

    """
    error_vector = err_vec2_to_err_vec(error_vector)
    sum = np.abs(error_vector[0])
    for index in range(1, len(error_vector)):
        sum += np.abs(error_vector[index] + error_vector[index-1])

    return sum*Ts/2


def itae(error_vector: np.array, Ts=0.01) -> float:
    """
    Integral of time multiplied Absolute Error Criterion
    (Criterio de la integral del valor absoluto del error
    multiplicado por el tiempo)

    """
    error_vector = err_vec2_to_err_vec(error_vector)
    sum = 0
    for index in range(1, len(error_vector)):
        sum += index*np.abs(error_vector[index] + error_vector[index-1])

    return sum*Ts/2


def err_vec2_to_err_vec(err_vec2: np.array) -> np.array:
    return np.sqrt(err_vec2[:, 0]**2 + err_vec2[:, 1]**2)


if __name__ == '__main__':
    err_vector = np.array([0.01, 0.2, 2])
    print(ise(err_vector))
    print(itse(err_vector))
    print(iae(err_vector))
    print(itae(err_vector))
