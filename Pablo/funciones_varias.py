import numpy as np

def distancia(pos1, pos2):
    d = np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
    return d

def posicion(ang):
    if 0 <= ang < np.pi/2:
        x = np.cos(ang) 
        y = np.sin(ang)
    if np.pi/2 <= ang< -np.pi :
        x = np.cos(ang) 
        y = -np.sin(ang)
    if -np.pi <= ang < -np.pi/2 :
        x = -np.cos(ang)
        y = -np.sin(ang)  
    if -np.pi/2 <= ang < 0 :
        x = -np.cos(ang) 
        y = np.sin(ang) 
    return (x,y)

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
        gamma = np.arctan2(y,x)  # esta bien calculado para cualquier caso
        print(gamma, alpha)
        theta1R = gamma - alpha
        theta2R = np.pi - beta
        theta1L = gamma + alpha
        theta2L = -np.pi + beta
        posible = True 
        return (theta1R, theta2R), (theta1L, theta2L), posible

def control_referencia(thetaR, thetaL, theta):
    
    posR = posicion(thetaR[0])
    posL = posicion(thetaL[0])
    pos = posicion(theta)

    dR = distancia(posR, pos)
    dL = distancia(posL, pos)
    
    if dR>dL:
        theta_ref = thetaL
    else:
        theta_ref = thetaR
    
    return theta_ref

def angulo_real(cos, sin):
    if sin > 0:
        theta = np.arccos(cos)
    else:
        theta = 2*np.pi - np.arccos(cos)
    return theta

params = {'L1': 0.1, 'L2': 0.1}

a, b, posible = theta_ref12(params, -15, -15)  
