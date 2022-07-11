from lib.graficos import (PID_PATH_EXPORT, graficos_estados_todos,
                          LIQ_PATH_EXPORT, OSC_PATH_EXPORT)
from lib.util import gen_tiempo
import numpy as np

Ts = 0.01
test_steps = 1000
t = gen_tiempo(Ts, test_steps)

states = np.load(PID_PATH_EXPORT+"_state.npy")
theta_real_t_pid = states[:, 0:2]
theta_ref_t_pid = states[:, 2:4]

states = np.load(LIQ_PATH_EXPORT+"_state.npy")
theta_real_t_liq = states[:, 0:2]

states = np.load(OSC_PATH_EXPORT+"_state.npy")
theta_real_t_osc = states[:, 0:2]

graficos_estados_todos(t, theta_ref_t_pid, theta_real_t_pid,
                       theta_real_t_liq, theta_real_t_osc)
