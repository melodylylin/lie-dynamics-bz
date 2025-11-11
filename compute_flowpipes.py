import casadi as ca
from lie.SE23 import *
from flowpipe.inner_bound import *
from flowpipe.outer_bound import *
from bezier.bezier_planning import generate_path, derive_bezier7
from flowpipe.flowpipe import *

def get_flowpipes():

    bezier7 = derive_bezier7()
    # bc_t = np.array([
    #         [ # position
    #         [0, 0, 0],  
    #         [0, 0, 5],    
    #         [0, 0, 5], 
    #         [120, -3.65, 5],
    #         [160, -5, 5],
    #         [190, -7.5, 5],
    #         [229.7, -9.6, 4],
    #         [229.7, -9.6, 4],
    #         [229.7, -9.6, 2],
    #         ],
    #         [ # velocity
    #         [0, 0, 0.1],  
    #         [0, 0, 0], 
    #         [0, 0, 0],
    #         [1, -0.1, 0],
    #         [1, -0.1, 0],
    #         [1, -0.1, 0],
    #         [0.1, -0.1, 0],
    #         [0, 0, 0],  # wp0, x, y, z
    #         [0, 0, 0],
    #         ],
    #         [ # accel
    #         [0, 0, 0], 
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         ],
    #         [ # jerk
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0], 
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         ]])
    # bc_t = np.array([
    #         [ # position
    #         [0, 0, 0],  
    #         [0, 0, 2],    
    #         [0, 0, 5], 
    #         [120, -3.05, 5],
    #         [160, -5.2, 5],
    #         [229.7, -9.6, 4],
    #         [229.7, -9.6, 4],
    #         [229.7, -9.6, 2],
    #         ],
    #         [ # velocity
    #         [0, 0, 0.1],  
    #         [0, 0, 0.1], 
    #         [0, 0, 0],
    #         [2.2, -0.1, 0],
    #         [2.2, -0.1, 0],
    #         [2.2, -0.1, 0],
    #         [0, 0, 0],  # wp0, x, y, z
    #         [0, 0, 0],
    #         ],
    #         [ # accel
    #         [0, 0, 0], 
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         ],
    #         [ # jerk
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         ]])
    d = 2
    vel = 1
    dt = 2*d/vel
    bc_t = np.array([
        [ # position (x, y, z)
            [0, 0, 2], # 1
            [0, -d, 1.5], # 2
            [0, 0, 2], # 3
            [0, d, 2.5], # 4
            [0, 0, 2], # 5
            [0, 0, 2], # 6
        ],
        [ # velocity
            [0, 0, 0], # 1, stop
            [-vel, 0, 0], # 2, stop
            [vel/2, vel/2, 0], # 3, take turn at given vel
            [-vel, 0, 0], # 4, take turn at given vel
            [0, 0, 0], # 5, take turn at given vel
            [0, 0, 0], # 6, stop when reach origin
        ],
        [ # accel
            [0, 0, 0], # 1
            [0, 0, 0], # 2
            [0, 0, 0], # 3
            [0, 0, 0], # 4
            [0, 0, 0], # 5
            [0, 0, 0], # 6
        ],
        [ # jerk
            [0, 0, 0], # 1
            [0, 0, 0], # 2
            [0, 0, 0], # 3
            [0, 0, 0], # 4
            [0, 0, 0], # 5
            [0, 0, 0], # 6
        ]])
    
    # solve for bezier trajectories
    k = 10
    ref = generate_path(bc_t, k, 0.01, dt)
    ax = [np.min(ref['acc_x']), np.max(ref['acc_x'])]
    ay = [np.min(ref['acc_y']), np.max(ref['acc_y'])]
    az = [np.min(ref['acc_z']), np.max(ref['acc_z'])]
    omega1 = [np.min(ref['omega_1']), np.max(ref['omega_1'])]
    omega2 = [np.min(ref['omega_2']), np.max(ref['omega_2'])]
    omega3 = [np.min(ref['omega_3']), np.max(ref['omega_3'])]

    # Set disturbance here
    w1 = 2 * 0.0  #0.05 #0.75 # disturbance for translational (impact a)
    w2 = 1.25 # disturbance for angular (impact alpha)

    sol = find_omega_invariant_set(omega1, omega2, omega3) 

    # Initial condition
    P_omega = sol['P']
    e0_omega = np.array([0,0,0]) # initial error
    beta_omega = (e0_omega.T@P_omega@e0_omega) # initial Lyapnov value

    # find bound
    omegabound = omega_bound(omega1, omega2, omega3, w2, beta_omega) 

    sol_LMI = find_se23_invariant_set(ax, ay, az, omega1, omega2, omega3)
    print('finished computing LMI')

    # Initial condition
    e = np.array([0,0,0,0,0,0,0,0,0]) # initial error in Lie group (nonlinear)

    # transfer initial error to Lie algebra (linear)
    e0 = ca.DM(SE23Dcm.vee(SE23Dcm.log(SE23Dcm.matrix(e))))
    e0 = np.array([e0]).reshape(9,)
    ebeta = e0.T@sol_LMI['P']@e0
    # Calculate convex hull for flow pipes
    n = 25 # number of flow pipes
    flowpipes_traj, intervalhull_traj, nom_traj, t_vect = flowpipes_3d(ref, n, ebeta, w1, 2.8*omegabound, sol_LMI)
    # with open('flowpipes.npy', 'wb') as f:
    #     np.save(f, np.array(flowpipes_traj))
    return flowpipes_traj

def main(args=None):
    get_flowpipes()


if __name__ == "__main__":
    main()