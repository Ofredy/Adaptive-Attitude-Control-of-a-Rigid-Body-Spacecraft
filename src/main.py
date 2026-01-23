import numpy as np

import integrator, mrp_functions, helper_functions


############################# PID Control Example #############################

# using k to denote timestep for this problem
inertia_tensor = np.array([[ 100.0, 0.0, 0.0 ],
                           [ 0.0, 75.0, 0.0 ],
                           [ 0.0, 0.0, 80.0 ]])

k = 5 # Nm
k_integral = 0.005 # Nm
p = np.array([[ 10.0, 0.0, 0.0 ],
              [ 0.0, 10.0, 0.0 ],
              [ 0.0, 0.0, 10.0 ]])

dt = 0.01
f = 0.05

mrp_b_n_0 = np.array([ 0.1, 0.2, -0.1 ])
w_b_n_0 = np.array([ 3.0, 1.0, -2.0 ]) * np.pi / 180

unmodeled_torque = np.array([ 0.5, -0.3, 0.2 ])

def get_target_mrp_r_n(t):
    
    return np.array([ 0.2 * np.sin( f*t ), 0.3 * np.cos( f*t ), -0.3 * np.sin( f*t ) ])

def get_target_mrp_r_n_dot(t):

    return np.array([ 0.2 * f * np.cos( f*t ), -0.3 * f * np.sin( f*t ), -0.3 * f * np.cos( f*t ) ])

def get_target_w_r_n(t):

    # solving for mrp_r_n_k
    mrp_r_n_k = get_target_mrp_r_n(t)

    # solving for w_r_n
    mrp_r_n_dot_k = get_target_mrp_r_n_dot(t)

    mrp_r_n_norm = np.linalg.norm(mrp_r_n_k)
    mrp_r_n_tilde = helper_functions.get_tilde_matrix(mrp_r_n_k)

    return 4 * np.linalg.inv( ( 1 - mrp_r_n_norm**2 ) * np.eye(3) + 2 * mrp_r_n_tilde + 2 * np.outer(mrp_r_n_k, mrp_r_n_k) ) @ mrp_r_n_dot_k

def get_target_info(t, dt):

    # solving for mrp_r_n_k
    mrp_r_n_k = get_target_mrp_r_n(t)

    # solving for w_r_n
    w_r_n_k = get_target_w_r_n(t)

    # solving for  w_r_n_dot_k
    w_r_n_dot_k = ( w_r_n_k - get_target_w_r_n(t-dt) ) / dt

    return mrp_r_n_k, w_r_n_k, w_r_n_dot_k

def get_mrp_b_r(mrp_b_n_k, mrp_r_n_k):
    """
    Compute the MRP of frame B relative to frame R.

    Args:
        mrp_b_n_k: MRP of B relative to N.
        mrp_r_n_k: MRP of R relative to N.

    Returns:
        mrp_b_r: MRP of B relative to R.
    """
    return mrp_functions.mrp_addition(mrp_b_n_k, mrp_functions.invert_mrp(mrp_r_n_k))

######## this is the complete PID implementation, will use this as already dene example ########
def get_att_track_control_k(x_k, t):
    """
    Compute the control torque for attitude tracking.

    Args:
        x_k: Current state [MRP_B/N, w_B/N].
        t: Current time.

    Returns:
        Control torque vector (u).
    """
    # State decomposition
    mrp_b_n_k = x_k[:3]  # MRP_B/N
    w_b_n_k = x_k[3:6]   # Angular velocity_B/N (in B)
    state_sum = x_k[6:9]
    w_b_r_0 = x_k[9:]

    # Target info
    mrp_r_n_k, w_r_n_k, w_r_n_dot_k = get_target_info(t, dt)  # Target MRP and angular velocity

    # Step 1: Compute relative MRP (sigma_B/R)
    mrp_b_r_k = get_mrp_b_r(mrp_b_n_k, mrp_r_n_k)

    # Step 2: Transform target angular velocity and acceleration to Body frame
    dcm_b_r = mrp_functions.mrp_to_dcm(mrp_b_r_k)  # DCM from R to B
    w_r_n_b = dcm_b_r @ w_r_n_k                   # Transform w_R/N to Body frame
    w_r_n_dot_b = dcm_b_r @ w_r_n_dot_k           # Transform w_R/N_dot to Body frame

    # Step 3: Compute relative angular velocity (omega_B/R in Body frame)
    w_b_r_k = w_b_n_k - w_r_n_b

    # Step 4: Compute control torque
    control_torque = (
        -k * mrp_b_r_k                         # Proportional term (error in attitude)
        - ( p + k_integral * p @ inertia_tensor ) @ w_b_r_k # Damping term (relative angular velocity)
        - k * k_integral * p @ state_sum + k_integral * p @ inertia_tensor @ w_b_r_0 # integral term
        + inertia_tensor @ (w_r_n_dot_b - np.cross(w_b_n_k, w_r_n_b))  # Inertia term
    )

    return control_torque

######## Will need to refactor this to work with different controllers ########
def get_att_track_state_dot(x_k, t):

    # spacecraft
    mrp_b_n_k = x_k[:3]
    w_b_n_k = x_k[3:6]
    state_sum = x_k[6:]

    # target
    mrp_r_n_k, w_r_n_k, _ = get_target_info(t, dt)

    # solving for mrp_b_n_dot_k
    mrp_b_n_norm = np.linalg.norm(mrp_b_n_k)
    mrp_b_n_tilde = helper_functions.get_tilde_matrix(mrp_b_n_k)

    mrp_b_r_k = get_mrp_b_r(mrp_b_n_k, mrp_r_n_k)
    mrp_b_r_dcm = mrp_functions.mrp_to_dcm(mrp_b_r_k)

    w_b_r_k = w_b_n_k - mrp_b_r_dcm @ w_r_n_k

    mrp_b_n_dot = ( 1/4 ) * ( ( 1 - mrp_b_n_norm**2 ) * np.eye(3) + 2 * mrp_b_n_tilde + 2 * np.outer(mrp_b_n_k, mrp_b_n_k) ) @ w_b_n_k

    # solving for w_b_n_dot_k
    w_b_n_dot = np.linalg.inv(inertia_tensor) @ ( get_att_track_control_k(x_k, t) + unmodeled_torque ) 

    # tracking state_sum
    state_sum_dot = mrp_b_r_k

    return np.concatenate( [ mrp_b_n_dot, w_b_n_dot, state_sum_dot, np.array([ 0.0, 0.0, 0.0 ]) ])

def get_track_error_at_time(mrp_sum, time, dt=0.01):

    mrp_b_n_k = mrp_sum[int(time/dt)][:3]
    mrp_r_n_k = get_target_mrp_r_n(time)

    mrp_b_r = get_mrp_b_r(mrp_b_n_k, mrp_r_n_k)
    print(np.linalg.norm(mrp_b_r))

def get_mean_absolute_error(mrp_sum, time_span, dt=0.01):
    """
    Calculate the mean absolute error for the attitude tracking over a given time span.
    
    Args:
        mrp_sum: Array of MRPs and angular velocities at each time step (state history).
        time_span: Total simulation time in seconds.
        dt: Time step size.
    
    Returns:
        Mean absolute error (MAE) for attitude tracking.
    """
    num_steps = int(time_span / dt)  # Number of time steps
    error_sum = 0  # Initialize error accumulator
    
    for i in range(num_steps):
        current_time = i * dt
        mrp_b_n_k = mrp_sum[i][:3]  # Extract current MRP_B/N
        mrp_r_n_k = get_target_mrp_r_n(current_time)  # Get target MRP_R/N at current time
        
        # Compute relative MRP (MRP_B/R)
        mrp_b_r = get_mrp_b_r(mrp_b_n_k, mrp_r_n_k)
        error_sum += np.linalg.norm(mrp_b_r)  # Add the norm of the relative MRP
    
    mean_absolute_error = error_sum / num_steps  # Compute the mean of the errors
    return mean_absolute_error


mrp_b_r_0 = get_mrp_b_r(mrp_b_n_0, get_target_mrp_r_n(0))
mrp_b_r_dcm = mrp_functions.mrp_to_dcm(mrp_b_r_0)
w_b_r_0 = w_b_n_0 - mrp_b_r_dcm @ get_target_w_r_n(0)

x_0 = np.concatenate([mrp_b_n_0, w_b_n_0, np.zeros(3), w_b_r_0])

mrp_sum = integrator.runge_kutta(get_att_track_state_dot, x_0, 0, 120, is_mrp=True, dt=dt)
get_track_error_at_time(mrp_sum, 45, dt=dt)