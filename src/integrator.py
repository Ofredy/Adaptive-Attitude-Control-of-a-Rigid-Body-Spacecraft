import numpy as np

import mrp_functions


def integrator(func_x, x_0, t_0, t_f, dt=0.1, norm_value=False, is_mrp=False):

    state_summary = []

    x_n = x_0
    state_summary.append(x_n)
    t = t_0

    while t < t_f:

        f = func_x(x_n, t)

        x_n = x_n + f * dt

        if norm_value:
            x_n = x_n / np.linalg.norm(x_n)

        if is_mrp:

            if mrp_functions.mrp_is_long(x_n):
                print("mrp is long: %.4f" % (np.linalg.norm(x_n)))
                x_n = mrp_functions.get_shadow_mrp(x_n)

        state_summary.append(x_n)

        t += dt

    return state_summary

def runge_kutta(func, x_0, t_0, t_f, dt=0.01, norm_value=False, is_mrp=False):

    state_summary = []

    x_n = x_0
    state_summary.append(x_n)
    t = t_0

    while t < t_f:

        k1 = dt * func(x_n, t)
        
        k2 = dt * func(x_n + 0.5 * k1, t + 0.5 * dt)
        
        k3 = dt * func(x_n + 0.5 * k2, t + 0.5 * dt)
        
        k4 = dt * func(x_n + k3, t + dt)

        x_n = x_n + (k1 + 2*k2 + 2*k3 + k4) / 6

        if norm_value:
            x_n = x_n / np.linalg.norm(x_n)

        if is_mrp:

            if mrp_functions.mrp_is_long(x_n[:3]):
                x_n[:3] = mrp_functions.get_shadow_mrp(x_n[:3])

        state_summary.append(x_n)

        t += dt

    return state_summary