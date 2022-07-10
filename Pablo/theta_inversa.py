import numpy as np

def theta_ref12(params, x, y):
    L1 = params['L1']
    L2 = params['L2']
    
    # el brazo no es tan largo
    if np.sqrt(x**2 + y**2) > L1 + L2:
        posible = False
        print('Imposible de llegar')
        return 0, 0, posible
    
    # menor a lo que puede el brazo
    if np.sqrt(x**2 + y**2) < L1 - L2:
        posible = False
        print('Imposible de llegar')
        return 0, 0, posible

    else:
        beta = np.arccos((L1**2 + L2**2 - (x**2 + y**2))/(2*L1*L2))
        alpha = np.arccos( (x**2 + y**2 + L1**2 - L2**2 ) / (2* L1 * np.sqrt(x**2 + y**2)) )
        gamma = np.arctan2(y,x)
        print(gamma, alpha)
        theta1 = gamma - alpha
        theta2 = np.pi - beta
        posible = True 
        print(f'Theta1: {theta1}, Theta2: {theta2}')
        return theta1, theta2, posible


params = {'L1': 10, 'L2': 15}

a, b, posible = theta_ref12(params, 30, 5)  
