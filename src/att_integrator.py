# library imports
import numpy as np

# our imports
import integrator, helper_functions, mrp_functions
from performance_plotter import plot_inertia_estimate_history, plot_mrp_error_history


class AttIntegrator:

    def __init__(self, 
                 mrp_b_n_0, w_b_n_0,
                 unmodeled_torque, 
                 actual_inertia_tensor, estimated_inertia_tensor, 
                 k, k_integral, p, 
                 learn_inertia=False, gamma=0.01,
                 f=0.05,
                 out_dir='sim_results', total_time=120.0, dt=0.01):
        
        # intial conditions
        self.mrp_b_n_0 = mrp_b_n_0
        self.w_b_n_0 = w_b_n_0

        # unmodeled torque 
        self.unmodeled_torque = unmodeled_torque

        # inertia tensor properties
        self.actual_inertia_tensor = actual_inertia_tensor
        self.estimated_inertia_tensor = estimated_inertia_tensor

        # feedback gain terms
        self.k = k 
        self.k_integral = k_integral
        self.p = p

        # adaptive controller properties
        self.learn_inertia = learn_inertia
        self.gamma = gamma

        # reference generation frequency
        self.f = f

        # sim constants
        self.out_dir = out_dir
        self.total_time = total_time
        self.dt = dt

    def get_target_mrp_r_n(self, t):

        return np.array([ 0.2 * np.sin( self.f*t ), 0.3 * np.cos( self.f*t ), -0.3 * np.sin( self.f*t ) ])

    def get_target_mrp_r_n_dot(self, t):

        return np.array([ 0.2 * self.f * np.cos( self.f*t ), -0.3 * self.f * np.sin( self.f*t ), -0.3 * self.f * np.cos( self.f*t ) ])

    def get_target_w_r_n(self, t):

        # solving for mrp_r_n_k
        mrp_r_n_k = self.get_target_mrp_r_n(t)

        # solving for w_r_n
        mrp_r_n_dot_k = self.get_target_mrp_r_n_dot(t)

        mrp_r_n_norm = np.linalg.norm(mrp_r_n_k)
        mrp_r_n_tilde = helper_functions.get_tilde_matrix(mrp_r_n_k)

        return 4 * np.linalg.inv( ( 1 - mrp_r_n_norm**2 ) * np.eye(3) + 2 * mrp_r_n_tilde + 2 * np.outer(mrp_r_n_k, mrp_r_n_k) ) @ mrp_r_n_dot_k

    def get_target_info(self, t, dt):

        # solving for mrp_r_n_k
        mrp_r_n_k = self.get_target_mrp_r_n(t)

        # solving for w_r_n
        w_r_n_k = self.get_target_w_r_n(t)

        # solving for  w_r_n_dot_k
        w_r_n_dot_k = ( w_r_n_k - self.get_target_w_r_n(t-dt) ) / dt

        return mrp_r_n_k, w_r_n_k, w_r_n_dot_k

    def get_mrp_b_r(self, mrp_b_n_k, mrp_r_n_k):
        """
        Compute the MRP of frame B relative to frame R.

        Args:
            mrp_b_n_k: MRP of B relative to N.
            mrp_r_n_k: MRP of R relative to N.

        Returns:
            mrp_b_r: MRP of B relative to R.
        """
        return mrp_functions.mrp_addition(mrp_b_n_k, mrp_functions.invert_mrp(mrp_r_n_k))
    
    def excitation_torque(self, t, period=20.0,      # seconds between bursts
                                   burst=3.0,        # burst duration
                                   tau_amp=1,     # N*m amplitude
                                   f_hz=0.8):        # excitation frequency
        
        w = 2*np.pi*f_hz
        # time within the repeating cycle
        tc = t % period

        if tc < burst:
            # multi-frequency / irrational ratios help excitation
            return tau_amp * np.array([
                np.sin(w*tc),
                np.sin(1.37*w*tc + 0.6),
                np.sin(0.73*w*tc + 1.1),
            ])
        
        return np.zeros(3)

    def get_att_track_control_k(self, x_k, t):
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
        w_b_r_0 = x_k[9:12]

        # Target info
        mrp_r_n_k, w_r_n_k, w_r_n_dot_k = self.get_target_info(t, self.dt)  # Target MRP and angular velocity

        # Step 1: Compute relative MRP (sigma_B/R)
        mrp_b_r_k = self.get_mrp_b_r(mrp_b_n_k, mrp_r_n_k)

        # Step 2: Transform target angular velocity and acceleration to Body frame
        dcm_b_r = mrp_functions.mrp_to_dcm(mrp_b_r_k)  # DCM from R to B
        w_r_n_b = dcm_b_r @ w_r_n_k                   # Transform w_R/N to Body frame
        w_r_n_dot_b = dcm_b_r @ w_r_n_dot_k           # Transform w_R/N_dot to Body frame

        # Step 3: Compute relative angular velocity (omega_B/R in Body frame)
        w_b_r_k = w_b_n_k - w_r_n_b

        # Step 4: Compute control torque
        control_torque = (
            - self.k * mrp_b_r_k                         # Proportional term (error in attitude)
            - ( self.p + self.k_integral * self.p @ self.estimated_inertia_tensor ) @ w_b_r_k # Damping term (relative angular velocity)
            - self.k * self.k_integral * self.p @ state_sum + self.k_integral * self.p @ self.estimated_inertia_tensor @ w_b_r_0 # integral term
            + self.estimated_inertia_tensor @ (w_r_n_dot_b - np.cross(w_b_n_k, w_r_n_b))  # w_r_n correction term
            + helper_functions.get_tilde_matrix(w_b_n_k) @ (self.estimated_inertia_tensor @ w_b_n_k) # inertia term correction
        )

        return control_torque
    
    def get_theta_dot(self, w_b_n, w_b_n_dot, delta_w, mrp_e):
        # sliding/composite signal (needs to be nonzero to learn)
        lam = 1.0  # try 0.5 to 2.0
        s = delta_w + lam * mrp_e

        wx, wy, wz = w_b_n
        wdx, wdy, wdz = w_b_n_dot

        Y = np.array([
            [wdx,  wy*wz,  -wy*wz],
            [-wz*wx, wdy,   wz*wx],
            [wx*wy, -wx*wy, wdz],
        ])

        Gamma = self.gamma * np.eye(3)  # gamma is now a real gain, not divided by J^2
        theta_dot = -Gamma @ (Y.T @ s)

        return theta_dot

    def get_att_track_state_dot(self, x_k, t):

        # spacecraft
        mrp_b_n_k = x_k[:3]
        w_b_n_k = x_k[3:6]
        state_sum = x_k[6:12]

        # target
        mrp_r_n_k, w_r_n_k, _ = self.get_target_info(t, self.dt)

        # solving for mrp_b_n_dot_k
        mrp_b_n_norm = np.linalg.norm(mrp_b_n_k)
        mrp_b_n_tilde = helper_functions.get_tilde_matrix(mrp_b_n_k)

        mrp_b_r_k = self.get_mrp_b_r(mrp_b_n_k, mrp_r_n_k)
        mrp_b_r_dcm = mrp_functions.mrp_to_dcm(mrp_b_r_k)

        w_b_r_k = w_b_n_k - mrp_b_r_dcm @ w_r_n_k

        mrp_b_n_dot = ( 1/4 ) * ( ( 1 - mrp_b_n_norm**2 ) * np.eye(3) + 2 * mrp_b_n_tilde + 2 * np.outer(mrp_b_n_k, mrp_b_n_k) ) @ w_b_n_k

        # solving for w_b_n_dot_k
        w_b_n_tilde = helper_functions.get_tilde_matrix(w_b_n_k)     # \tilde{ω}
        w_b_n_dot = np.linalg.inv(self.actual_inertia_tensor) @ ( self.get_att_track_control_k(x_k, t) + self.unmodeled_torque - w_b_n_tilde @ (self.actual_inertia_tensor @ w_b_n_k) ) 

        # solving for theta_dot & updating estimate_inertia_tensor
        if self.learn_inertia:
            theta_dot = self.get_theta_dot(w_b_n=w_b_n_k, w_b_n_dot=w_b_n_dot, delta_w=w_b_r_k, mrp_e=mrp_b_r_k)
            self.estimated_inertia_tensor = np.diag(x_k[12:15])

        # tracking state_sum
        state_sum_dot = mrp_b_r_k

        x_dot_parts = [
                    mrp_b_n_dot,
                    w_b_n_dot,
                    state_sum_dot, 
                    np.array([ 0.0, 0.0, 0.0 ])
                ]

        if self.learn_inertia:
            x_dot_parts.append(theta_dot)

        if int(t / self.dt) % 100 == 0 and self.learn_inertia:
            print("||dw||", np.linalg.norm(w_b_r_k),
                  "||e||", np.linalg.norm(mrp_b_r_k),
                  "||theta_dot||", np.linalg.norm(theta_dot),
                  "Inertia Estimate diag:", np.diag(self.estimated_inertia_tensor))

        return np.concatenate( x_dot_parts )
    
    def get_track_error_at_time(self, mrp_sum, time, dt=0.01):

        mrp_b_n_k = mrp_sum[int(time/dt)][:3]
        mrp_r_n_k = self.get_target_mrp_r_n(time)

        mrp_b_r = self.get_mrp_b_r(mrp_b_n_k, mrp_r_n_k)
        print(np.linalg.norm(mrp_b_r))

    def compute_track_error_history(self, t_0=0.0, dt=None):
        """
        Computes sigma_{B/R}(t) for the whole sim using:
          mrp_b_n(t) from mrp_sum
          mrp_r_n(t) from self.get_target_mrp_r_n(t)
          sigma_b_r(t) = self.get_mrp_b_r(mrp_b_n, mrp_r_n)

        Returns:
          mrp_err_hist: (N,3)
          t_hist: (N,)
        """
        if dt is None:
            dt = self.dt

        X = np.asarray(self.mrp_sum)
        # support (N,nx) or (nx,N)
        if X.ndim != 2:
            raise ValueError("mrp_sum must be 2D.")
        # if it looks like (nx, N), transpose to (N, nx)
        if X.shape[0] in (18, 6, 12) and X.shape[1] != X.shape[0]:
            X = X.T

        N = X.shape[0]
        t_hist = t_0 + np.arange(N) * dt
        self.mrp_err_hist = np.zeros((N, 3))

        for i, t in enumerate(t_hist):
            mrp_b_n = X[i, 0:3]
            mrp_r_n = self.get_target_mrp_r_n(t)
            self.mrp_err_hist[i, :] = self.get_mrp_b_r(mrp_b_n, mrp_r_n)

    def get_mean_absolute_error(self, mrp_sum, time_span, dt=0.01):
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
            mrp_r_n_k = self.get_target_mrp_r_n(current_time)  # Get target MRP_R/N at current time

            # Compute relative MRP (MRP_B/R)
            mrp_b_r = self.get_mrp_b_r(mrp_b_n_k, mrp_r_n_k)
            error_sum += np.linalg.norm(mrp_b_r)  # Add the norm of the relative MRP

        mean_absolute_error = error_sum / num_steps  # Compute the mean of the errors
        return mean_absolute_error
    
    def simulate_system(self):

        mrp_b_r_0 = self.get_mrp_b_r(self.mrp_b_n_0, self.get_target_mrp_r_n(0))
        dcm_b_r_0 = mrp_functions.mrp_to_dcm(mrp_b_r_0)
        w_b_r_0   = self.w_b_n_0 - dcm_b_r_0 @ self.get_target_w_r_n(0)

        x_parts = [
            self.mrp_b_n_0,
            self.w_b_n_0,
            np.zeros(3),
            w_b_r_0
        ]

        if self.learn_inertia:
            theta_0 = np.diag(self.estimated_inertia_tensor)
            x_parts.append(theta_0)

        x_0 = np.concatenate(x_parts)

        self.mrp_sum = integrator.runge_kutta(self.get_att_track_state_dot, x_0, 0, self.total_time, is_mrp=True, dt=self.dt)
        self.get_track_error_at_time(self.mrp_sum, 45, dt=self.dt)

        self.compute_track_error_history()

        plot_mrp_error_history(
                                   mrp_err_hist=np.asarray(self.mrp_err_hist)[:, 0:3],
                                   dt=self.dt,
                                   out_dir=self.out_dir if hasattr(self, "output_dir") else ".",
                                   prefix="att_track",
                               )

        if self.learn_inertia:
            plot_inertia_estimate_history(
                mrp_sum=self.mrp_sum,
                dt=self.dt,
                J_actual=self.actual_inertia_tensor,
                out_dir=self.out_dir if hasattr(self, "output_dir") else ".",
                theta_start_idx=12,
                prefix="att_track",
                units="kg*m^2",
            )
