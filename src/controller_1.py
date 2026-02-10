# library imports
import numpy as np

from att_integrator import AttIntegrator


################ PD Controller + Adaptive Inertia Tensor Estimation ################

# inertia tensor
actual_inertia_tensor = np.array([[ 100.0, 0.0, 0.0 ],
                                  [ 0.0, 75.0, 0.0  ],
                                  [ 0.0, 0.0, 80.0  ]])

estimated_inertia_tensor = np.array([[ 100.0, 0.0, 0.0 ],
                                     [ 0.0, 75.0, 0.0  ],
                                     [ 0.0, 0.0, 80.0  ]])

# feedback gains
k = 5.0 # Nm
k_integral = 0.005 # Nm
p = np.array([[ 10.0, 0.0, 0.0 ],
              [ 0.0, 10.0, 0.0 ],
              [ 0.0, 0.0, 10.0 ]])

# intial conditions
mrp_b_n_0 = np.array([ 0.1, 0.2, -0.1 ])
w_b_n_0 = np.array([ 3.0, 1.0, -2.0 ]) * np.pi / 180

# unmodeled torque properties
unmodeled_torque = np.array([ 0.5, -0.3, 0.2 ])

# creating simulation env class & running the sim
pd_controller = AttIntegrator(mrp_b_n_0=mrp_b_n_0, w_b_n_0=w_b_n_0,
                              unmodeled_torque=unmodeled_torque,
                              actual_inertia_tensor=actual_inertia_tensor, estimated_inertia_tensor=estimated_inertia_tensor,
                              learn_inertia=False, gamma=0.0,
                              k=k, k_integral=k_integral, p=p,
                              total_time=500)

pd_controller.simulate_system()
