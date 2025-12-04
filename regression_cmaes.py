import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import cma
import warnings

from utils import config_old

WORKING_DIR = Path.cwd()
DATA_DIR = WORKING_DIR / 'data'
save_fig = False
g0 = 9.80665

warnings.filterwarnings('ignore', category=RuntimeWarning)


def compute_thrust(coeffs, f):
    """T = m*f + c"""
    return coeffs[0]*f + coeffs[1]


def compute_angle_attack(dihedral, yaw_servo_angle, wing="R"):
    alpha = 0
    lw = config_old.FLAPPER_DIMS["lw"]
    ly = config_old.FLAPPER_DIMS["ly"]
    lk = config_old.FLAPPER_DIMS["lk"]

    if wing == "R":
        arg = (- lw * np.sin(dihedral) - ly * np.sin(yaw_servo_angle)) / lk
        # Clamp to valid arcsin range
        arg = np.clip(arg, -1.0, 1.0)
        alpha = np.arcsin(arg)
    elif wing == "L":
        arg = (- lw * np.sin(dihedral) + ly * np.sin(yaw_servo_angle)) / lk
        # Clamp to valid arcsin range
        arg = np.clip(arg, -1.0, 1.0)
        alpha = np.arcsin(arg)
    else:
        raise ValueError('Only two wings, choose "R" or "L"')

    return alpha


def pitch_dynamics(t, state, interp_funcs, thrust_coeffs, k_params, neutral_pos):
    """
    ODE system for pitch dynamics: [theta, q]
    """
    theta, q = state
    
    # Safety check for state
    if not np.isfinite(theta) or not np.isfinite(q):
        return [0.0, 0.0]
    
    # Constrain pitch rate to reasonable bounds
    q = np.clip(q, -10.0, 10.0)
    
    # Get interpolated values at current time
    try:
        u = float(interp_funcs['u'](t))
        w = float(interp_funcs['w'](t))
        p = float(interp_funcs['p'](t))
        freq_L = float(interp_funcs['freq_L'](t))
        freq_R = float(interp_funcs['freq_R'](t))
        dihedral = float(interp_funcs['dihedral'](t))
        grad_dihedral = float(interp_funcs['grad_dihedral'](t))
    except Exception:
        return [0.0, 0.0]
    
    # Physical parameters
    lz = config_old.FLAPPER_DIMS["lz"]
    lw = config_old.FLAPPER_DIMS["lw"]
    Iyy = config_old.MMOI_WITH_WINGS_XY["Iyy"]
    
    # Compute angle of attack
    alpha_L = compute_angle_attack(dihedral, 0, "L")
    alpha_R = compute_angle_attack(dihedral, 0, "R")
    
    # Compute thrust
    thrust_left = compute_thrust(thrust_coeffs, freq_L)
    thrust_right = compute_thrust(thrust_coeffs, freq_R)
    
    # Compute vertical velocity components
    sin_dih = np.sin(dihedral)
    cos_dih = np.cos(dihedral)
    z_L = freq_L * (w - lw * sin_dih * q + lw * cos_dih * p)
    z_R = freq_R * (w - lw * sin_dih * q - lw * cos_dih * p)
    
    # Compute moment terms
    k_M1, k_M2, k_M3, k_M4 = k_params
    
    sin_dih_delta = np.sin(dihedral - neutral_pos)
    
    # Compute each term separately with bounds checking
    term1 = -(freq_L + freq_R) * (u - lz * q + lw * grad_dihedral * sin_dih_delta) * lz
    term2 = (thrust_left * np.sin(alpha_L) + thrust_right * np.sin(alpha_R)) * lz
    term3 = -(z_L + z_R) * lw * sin_dih_delta
    term4 = (thrust_right + thrust_left) * lw * sin_dih_delta
    
    # Apply coefficients
    k_M1_term = k_M1 * term1
    k_M2_term = k_M2 * term2
    k_M3_term = k_M3 * term3
    k_M4_term = k_M4 * term4
    
    # Total moment with clipping to prevent numerical overflow
    M_total = k_M1_term + k_M2_term + k_M3_term + k_M4_term
    
    # Check for invalid values
    if not np.isfinite(M_total):
        return [0.0, 0.0]
    
    # Clip to reasonable moment bounds (adjust based on your system)
    M_total = np.clip(M_total, -50.0, 50.0)
    
    # Pitch acceleration
    q_dot = M_total / Iyy
    q_dot = np.clip(q_dot, -50.0, 50.0)
    
    # State derivatives
    theta_dot = q
    
    return [float(theta_dot), float(q_dot)]


def cost_function(k_params, time, theta_measured, q0, theta0, interp_funcs, thrust_coeffs, neutral_pos, verbose=False):
    """
    Cost function for CMA-ES: RMS error between predicted and measured pitch angle
    """
    # Check for reasonable parameter values
    if np.any(np.abs(k_params) > 100):
        if verbose:
            print(f"  Params out of bounds: {k_params}")
        return 1e10
    
    try:
        # Solve ODE with stricter tolerances and max steps
        sol = solve_ivp(
            pitch_dynamics,
            [time[0], time[-1]],
            [theta0, q0],
            args=(interp_funcs, thrust_coeffs, k_params, neutral_pos),
            t_eval=time,
            method='RK45',
            rtol=1e-5,
            atol=1e-8,
            max_step=0.01  # Limit step size for stability
        )
        
        if not sol.success:
            if verbose:
                print(f"  Integration failed: {sol.message}")
            return 1e10
        
        theta_predicted = sol.y[0]
        
        # Check for reasonable output
        if not np.all(np.isfinite(theta_predicted)):
            if verbose:
                print(f"  Non-finite values in prediction")
            return 1e10
        
        # Check if arrays match in length
        if len(theta_predicted) != len(theta_measured):
            if verbose:
                print(f"  Length mismatch: {len(theta_predicted)} vs {len(theta_measured)}")
            return 1e10
        
        # Compute RMS error
        error = np.sqrt(np.mean((theta_predicted - theta_measured)**2))
        
        # Add regularization to prevent extreme parameters
        reg = 0.001 * np.sum(k_params**2)
        
        if verbose:
            print(f"  Success! RMSE={error:.6f}, params={k_params}")
        
        return error + reg
        
    except Exception as e:
        if verbose:
            print(f"  Exception: {type(e).__name__}: {e}")
        return 1e10


def regression_longitudinal_cmaes(thrust_coeffs, k_initial=None):
    """
    Use CMA-ES to fit longitudinal moment coefficients by directly matching pitch angle trajectory
    
    Args:
        thrust_coeffs: Thrust model coefficients [c1, c2] from vertical regression
        k_initial: Optional initial guess [k_M1, k_M2, k_M3, k_M4] from least squares
    """
    dataframes = {
        "longitudinal1": slice(720, 5229),
    }

    columns = [
        "time", "optitrack.freq.left", "optitrack.freq.right", 
        "optitrack.acc.x", "optitrack.vel.x", "optitrack.vel.z", 
        "optitrack.dihedral.left", "optitrack.dihedral.right", 
        "optitrack.q", "optitrack.pitch", "optitrack.roll", "optitrack.p"
    ]

    print("=" * 70)
    print("\nCMA-ES Regression for longitudinal moments\n")
    print("Optimizing coefficients by fitting pitch angle trajectory\n")

    # Load and process dataframe
    dfs = []
    for data_name, row_slice in dataframes.items():
        file_path = DATA_DIR / data_name / f"{data_name}-processed.csv"
        df = pd.read_csv(file_path, usecols=columns).iloc[row_slice].copy()
        
        df["dihedral"] = df["optitrack.dihedral.right"]
        df["grad_dihedral"] = np.gradient(df["dihedral"], df["time"])
        
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

    # Extract time series
    time = combined_df["time"].values
    time = time - time[0]  # Start from t=0
    
    u = combined_df["optitrack.vel.x"].values
    w = combined_df["optitrack.vel.z"].values
    p = combined_df["optitrack.p"].values
    q = combined_df["optitrack.q"].values
    theta = combined_df["optitrack.pitch"].values
    freq_L = combined_df["optitrack.freq.left"].values
    freq_R = combined_df["optitrack.freq.right"].values
    dihedral = combined_df["dihedral"].values
    grad_dihedral = combined_df["grad_dihedral"].values
    
    # Create interpolation functions for all inputs
    interp_funcs = {
        'u': interp1d(time, u, kind='linear', bounds_error=False, fill_value=(u[0], u[-1])),
        'w': interp1d(time, w, kind='linear', bounds_error=False, fill_value=(w[0], w[-1])),
        'p': interp1d(time, p, kind='linear', bounds_error=False, fill_value=(p[0], p[-1])),
        'freq_L': interp1d(time, freq_L, kind='linear', bounds_error=False, fill_value=(freq_L[0], freq_L[-1])),
        'freq_R': interp1d(time, freq_R, kind='linear', bounds_error=False, fill_value=(freq_R[0], freq_R[-1])),
        'dihedral': interp1d(time, dihedral, kind='linear', bounds_error=False, fill_value=(dihedral[0], dihedral[-1])),
        'grad_dihedral': interp1d(time, grad_dihedral, kind='linear', bounds_error=False, fill_value=(grad_dihedral[0], grad_dihedral[-1])),
    }
    
    # Initial conditions
    theta0 = theta[0]
    q0 = q[0]
    
    # Neutral dihedral position
    neutral_pos = 0.187
    
    # Initial guess for k_params
    if k_initial is not None:
        x0 = k_initial
        print(f"Using provided initial guess: k_M1={x0[0]:.6f}, k_M2={x0[1]:.6f}, k_M3={x0[2]:.6f}, k_M4={x0[3]:.6f}")
    else:
        # Default: use typical least squares values
        x0 = [-0.0001, -0.0005, -0.002, 0.04]
        print(f"Using default initial guess: k_M1={x0[0]:.6f}, k_M2={x0[1]:.6f}, k_M3={x0[2]:.6f}, k_M4={x0[3]:.6f}")
    
    # CMA-ES optimization
    print("Starting CMA-ES optimization...")
    
    sigma0 = 0.01  # Small initial standard deviation for local refinement
    
    options = {
        'popsize': 16,
        'maxiter': 100,
        'tolx': 1e-9,
        'tolfun': 1e-9,
        'verb_disp': 5,
        'verb_log': 0,
        'bounds': [[-0.1, -0.1, -0.1, -0.1], [0.1, 0.1, 0.1, 0.1]],  # Much tighter bounds
        'CMA_stds': [0.01, 0.01, 0.01, 0.01]  # Small exploration around LS solution
    }
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, options)
    
    iteration = 0
    best_valid_cost = 1e10
    best_valid_params = x0
    
    while not es.stop():
        solutions = es.ask()
        fitness = []
        
        for k in solutions:
            cost = cost_function(k, time, theta, q0, theta0, interp_funcs, thrust_coeffs, neutral_pos)
            fitness.append(cost)
            
            # Track best valid solution
            if cost < best_valid_cost and cost < 1e9:
                best_valid_cost = cost
                best_valid_params = k.copy()
        
        es.tell(solutions, fitness)
        
        if iteration % 5 == 0:
            print(f"Iteration {iteration}: Best cost = {min(fitness):.6f}")
        iteration += 1
    
    # Use best valid solution found
    if best_valid_cost < 1e9:
        k_opt = best_valid_params
        best_cost = best_valid_cost
    else:
        print("\nWarning: No valid solution found, using initial guess")
        k_opt = x0
        best_cost = cost_function(k_opt, time, theta, q0, theta0, interp_funcs, thrust_coeffs, neutral_pos)
    
    print("\n" + "=" * 70)
    print("Optimization complete!")
    print(f"Final RMSE: {best_cost:.6f} radians ({np.degrees(best_cost):.4f} degrees)")
    print(f"\nOptimized coefficients:")
    print(f"  k_M1 = {k_opt[0]:.6f}")
    print(f"  k_M2 = {k_opt[1]:.6f}")
    print(f"  k_M3 = {k_opt[2]:.6f}")
    print(f"  k_M4 = {k_opt[3]:.6f}")
    
    # Simulate with optimized parameters
    sol = solve_ivp(
        pitch_dynamics,
        [time[0], time[-1]],
        [theta0, q0],
        args=(interp_funcs, thrust_coeffs, k_opt, neutral_pos),
        t_eval=time,
        method='RK45',
        rtol=1e-5,
        atol=1e-8,
        max_step=0.01
    )
    
    if not sol.success:
        print("Warning: Final simulation failed")
        return k_opt, 0.0, best_cost
    
    theta_predicted = sol.y[0]
    q_predicted = sol.y[1]
    
    # Compute R² for pitch angle
    ss_res = np.sum((theta - theta_predicted)**2)
    ss_tot = np.sum((theta - np.mean(theta))**2)
    r2_theta = 1 - (ss_res / ss_tot)
    
    print(f"R² for pitch angle: {r2_theta:.4f}")
    
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot pitch angle
    axes[0].plot(time, theta, label='Measured θ', alpha=0.7, linewidth=2)
    axes[0].plot(time, theta_predicted, label='Predicted θ (CMA-ES)', linestyle='--', linewidth=2)
    axes[0].set_ylabel('Pitch angle θ (rad)')
    axes[0].set_title(f'Pitch Angle Fit (R² = {r2_theta:.4f}, RMSE = {np.degrees(best_cost):.4f}°)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot pitch rate
    axes[1].plot(time, q, label='Measured q', alpha=0.7, linewidth=2)
    axes[1].plot(time, q_predicted, label='Predicted q (CMA-ES)', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Pitch rate q (rad/s)')
    axes[1].set_title('Pitch Rate Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig("outputs/longitudinal_cmaes_regression.png", dpi=300)
    
    # Plot error over time
    fig, ax = plt.subplots(figsize=(12, 4))
    error_deg = np.degrees(theta - theta_predicted)
    ax.plot(time, error_deg, linewidth=1.5)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(time, error_deg, alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (degrees)')
    ax.set_title('Pitch Angle Prediction Error')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_fig:
        plt.savefig("outputs/longitudinal_cmaes_error.png", dpi=300)
    
    print("=" * 70)
    
    return k_opt, r2_theta, best_cost


if __name__ == "__main__":
    # Import the regression functions
    import sys
    sys.path.append(str(WORKING_DIR))
    from regression import regression_vertical_forces, regression_longitudinal_forces
    
    # Get thrust coefficients from vertical regression
    print("Running vertical regression to get thrust coefficients...")
    thrust_coeffs, _ = regression_vertical_forces()
    
    # Get least squares moment coefficients as initial guess
    print("\nRunning least squares longitudinal regression for initial guess...")
    coeffs_ls, r2_ls = regression_longitudinal_forces(thrust_coeffs)
    
    print(f"\nLeast squares gave: k_M1={coeffs_ls[0]:.6f}, k_M2={coeffs_ls[1]:.6f}, "
          f"k_M3={coeffs_ls[2]:.6f}, k_M4={coeffs_ls[3]:.6f}")
    print(f"Least squares R² for moments: {r2_ls:.4f}")
    
    # Run CMA-ES optimization starting from least squares solution
    print("\n" + "="*70)
    print("Now running CMA-ES to refine coefficients by fitting pitch angle...")
    print("="*70 + "\n")
    
    k_params, r2, rmse = regression_longitudinal_cmaes(thrust_coeffs, k_initial=coeffs_ls)
    
    print("\n" + "="*70)
    print("COMPARISON:")
    print("="*70)
    print(f"Least Squares (on moments):  R² = {r2_ls:.4f}")
    print(f"CMA-ES (on pitch angle):     R² = {r2:.4f}, RMSE = {np.degrees(rmse):.2f}°")
    print("="*70)
    
    plt.show()