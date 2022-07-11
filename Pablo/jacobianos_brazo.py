import numpy as np 

def jacobianos_robot(obs, tau, l1= 0.1, l2=0.1, m= 0.03):
    #HAY QUE CAMBIAR ESTAS LINEAS
    comun_den = (l1*(16*l1**2 + 12*l1*l2*obs[1] + 3*l2**2*(17-12*obs[1]**2)))
    A33_num = 12*l2*obs3*(6*l1*l2*obs1*obs6 + l1**2*obs7 + 3*l1**2*(obs6 + obs7))
    A34_num = 12*l2*(l1**2 + 3*l2**2)*obs3*(obs6 + obs7)
    A43_num = -(12*l2*obs[3]*(16*l1**2*obs[6]+ 3*l2**2*(obs[6]+obs[7]) + 6*l1*l2*obs[1]*(2*obs[6] + obs[7])))
    A44_num = -(36*l2**2 *(l2 + 2*l1*obs[1])*obs[3]*(obs[6] + obs[7]))

    A33 = A33_num/comun_den
    A34 = A34_num/comun_den
    A43 = A43_num/comun_den
    A44 = A44_num/comun_den

    # Hasta aqui debería converger


    A32_1 = (6*l2*(-2*(l1-6*l2*obs[1])*obs[3]*(-6*(l1**2+3*l2**2)*m*(l1*l2*m*obs[3]*obs[7]*(2*obs[6]+obs[7]) + 2*tau[0]) - 18*l2*m(l2+2*l1*obs[1])*
    (l1*l2*m*obs[3]*obs[6]**2 - 2*tau[1])) + l1*m*(16*l1**2 +33*l2**2 + 6*l2*(2*l1*obs[1] - 3*l2*COS2Q2))*m*(3* l2**2 * obs[1]*(obs[6] + obs[7])**2 + 
    l1**2 * obs[1]*obs[7]*(2*obs[6] + obs[7]) + 6*l1*l2*obs[6]**2 * COS2Q2 ) + 12*obs[3]*tau[1]))
    
    A32_2 = (l1**2 * m**2 *(16*l1**2 + 33*l2**2 + 6*l2*(2*l1+obs[1] - 3*l2*COS2Q2))**2)
    
    A32 = A32_1/A32_2
    A33 = -(12*l2*obs[3]*(6*l1*l2*obs[1]*obs[6] + l1**2 * obs[7] + 3*l2**2*(obs[6] + obs[7])))/(l1*(16*l1**2 + 33*l2**2 + 12*l1*l2*obs[1] - 18*l2**2 * COS2Q2))

    A34 = -(36*l2**2*(l2*+2*l1*obs[1])*obs[3]*(obs[6]+obs[7])) / (l1*(16*l1**2 + 33*l2**2 + 6*l2*(2*l1*obs[1] - 3*l2*COS2Q2)))
    
    A42 = (6*l2*(l1*m*(-256*l1**4 *obs[1]*obs[6]**2 + 9*l2**4*obs[1]*(-17 + 12*obs[1]**2 + 24*obs[3]**2)*(obs[6] + obs[7])**2 + 
        24*l1**2 * l2**2 * obs[1]*(6*(-6 + 3*obs[1]**2 + 8* obs[3]**2)*obs[6]**2 - 2*(2 + 3*obs[1]**2)*obs[6]*obs[7] - (2 + 3*obs[1]**2)*obs[7]**2)
        -96*l1**3 * l2* (-obs[3]**2 * obs[7]*(2*obs[6] + obs[7]) + obs[1]**2 * (4*obs[6]**2)+ 2* obs[6]*obs[7] + obs[7]**2)) + 18*l1*l2**3*
        (12*obs[1]**4*(2*obs[6]**2 + 2*obs[6]*obs[7] + obs[7]**2) + obs[3]**2*(32*obs[6]**2 + 30*obs[6]*obs[7] + 15 *obs[7]**2) +
        obs[1]**2*(12*(-3 + 2*obs[3]**2)*obs[6]**2 + 2*(-19 + 12*obs[3]**2)*obs[6]*obs[7] + (-19 + 12*obs[3]**2)*obs[6]*obs[7] +
        (-19 + 12*obs[3]**2)*obs[7]**2))) + 12*(16*l1**3 + 36*l2**3*obs[1] + 9*l1*l2**2*(5+4*obs[1]**2))*obs[3]*tau[0] - 144*l2*(l2 + 2*l1*obs[1])*(8*l1 + 3*l2*obs[1])*
        obs[3]*tau[1]) / (l1**2*m*(16*l1**2 + 12*l1*l2*obs[1] + 3*l2**2*(17 - 12*obs[1]**2))**2)

    A43 = -(12*l2*obs[3]*(16*l1**2*obs[6]+ 3*l2**2*(obs[6]+obs[7]) + 6*l1*l2*obs[1]*(2*obs[6] + obs[7]))) / (l1*(16*l1**2 + 12*l1*l2*obs[1] + 3*l2**2*(17-12*obs[1]**2)))

    A44 = - (36*l1**2 *(l2 + 2*l1*obs[1])*obs[3]*(obs[6] + obs[7])) / (l1*(16*l1**2 + 12*l1*l2*obs[1] + 3*l2**2*(17-12*obs[1]**2)))

    A = np.array([  [0,0,1,0],
                    [0,0,0,1],
                    [0,A32 ,A33 ,A34],
                    [0, A42, A43, A44]
                ])



    return A


