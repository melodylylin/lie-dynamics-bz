import numpy as np
import picos
import control
import itertools
import scipy
from lie.SE23 import *
from lie.se3 import *
import cyecca.lie as lie
from cyecca.lie.group_so3 import so3
from cyecca.lie.group_se23 import se23
import matplotlib.pyplot as plt
import cvxpy as cp
from sim.multirotor_ca_sim import simulate_multirotor
from bezier.bezier_planning import *

bezier7 = derive_bezier7()

def solve_lmi(alpha, A_list, verbosity=False):
    n = 9 # Augmented state dim
    P = cp.Variable((n,n), 'P', PSD=True)
    P1 = P[:3, :]
    P2 = P[3:6, :]
    P3 = P[6:, :]
    mu1 = cp.Variable((1,1), 'mu_1')
    mu2 = cp.Variable((1,1), 'mu_2')
    mu3 = cp.Variable((1,1), 'mu_3')
    gamma = mu1 + mu2 + mu3
    # gamma = cp.Variable((1,1), 'gamma')
    constraints = [P>>np.eye(9)]

    for Ai in A_list:
        constraints += [cp.bmat([[Ai.T*P + P*Ai + alpha*P, P],
                                [P1, -alpha*mu1*np.eye(3), np.zeros((3,3)), np.zeros((3,3))],
                                [P2, np.zeros((3,3)), -alpha*mu2*np.eye(3), np.zeros((3,3))],
                                [P3, np.zeros((3,3)), np.zeros((3,3)), -alpha*mu3*np.eye(3)]]) << 0] 
    prob = cp.Problem(cp.Minimize(gamma), constraints) 
    try:
        prob.solve()
        cost = gamma.value
        return {
            'cost': cost,
            'prob': prob,
            'mu1': mu1.value,
            'mu2': mu2.value,
            'mu3': mu3.value,
            'P': np.array(P.value),
            'alpha':alpha
            }

    except Exception as e:
        print(e)
        cost = -1    
        return {
            'cost': cost,
        }   

def se23_solve_control(ax,ay,az,omega1,omega2,omega3):
    A = -ca.DM(SE23Dcm.ad_matrix(np.array([0,0,0,ax,ay,az,omega1,omega2,omega3])))
    B = np.array([[0,0,0,0], # vx
                  [0,0,0,0], # vy
                  [0,0,0,0], # vz
                  [0,0,0,0], # ax
                  [0,0,0,0], # ay
                  [1,0,0,0], # az
                  [0,1,0,0], # omega1
                  [0,0,1,0], # omega2
                  [0,0,0,1]]) # omega3 # control omega1,2,3, and az
    Q = 10*np.eye(9)  # penalize state
    R = 1*np.eye(4)  # penalize input
    K, _, _ = control.lqr(A, B, Q, R) 
    K = -K # rescale K, set negative feedback sign
    BK = B@K
    return B, K, BK , A+B@K

def find_se23_invariant_set(ax,ay,az,omega1,omega2,omega3, verbosity=0):

    A = -np.array(ca.DM(lie.se23.elem(ca.DM([0, 0, 0, 0, 0, 9.8, 0, 0, 0])).ad()+SE23Dcm.adC_matrix()))
    #A = -ca.DM(se23.elem(ca.vertcat(0, 0, 0, 0, 0, 9.8, 0, 0, 0)).ad() + adC_matrix())
    B = np.eye(9)
    Q = 8*np.eye(9)
    R = np.eye(9)
    K, _, _ = control.lqr(A, B, Q, R)

    A_list = []
    for x0 in [0]:
        for x1 in [0]:
            for x2 in [0]:
                for x3 in ax:
                    for x4 in ay:
                        for x5 in az:
                            for x6 in omega1:
                                for x7 in omega2:
                                    for x8 in omega3:
                                        Ai = -np.array(ca.DM(lie.se23.elem(ca.DM([x0, x1, x2, x3, x4, x5, x6, x7, x8])).ad()+SE23Dcm.adC_matrix())) - B@K
                                        A_list.append(Ai)
    eig=[]
    for A in A_list:
        eig.append(np.linalg.eig(A)[0])

    if verbosity > 0:
        print('line search')
    # we perform a line search over alpha to find the feasible solution for LMIs
    print(-np.real(np.max(eig)))
    # alpha_opt = scipy.optimize.fminbound(lambda alpha: solve_lmi(alpha, A_list, verbosity=verbosity)['cost'], x1=0.001, x2=-np.real(np.max(eig)), disp=True if verbosity > 0 else False)
        
    sol = solve_lmi(1, A_list)
        
    return sol

def se23_invariant_set_points_theta(sol, t, w1_norm, w2_norm, beta): # w1_norm: a # w2_norm: omega  
    val = np.real(beta*np.exp(-sol['alpha']*t) + (sol['mu2']*w1_norm**2 + sol['mu3']*w2_norm**2)*(1-np.exp(-sol['alpha']*t))) # V(t)
    # 1 = xT(P/V(t))x, equation for the ellipse
    P1 = sol['P']/val
    A1 = P1[:6,:6]
    B1 = P1[:6,6:]
    C1 = P1[6:,:6]
    D1 = P1[6:,6:]
    P = D1-C1@np.linalg.inv(A1)@B1
    
    evals, evects = np.linalg.eig(P)
    radii = 1/np.sqrt(evals)
    R = evects@np.diag(radii)
    R = np.real(R)
    
    # draw sphere
    points = []
    n = 30
    for u in np.linspace(0, 2*np.pi, n):
        for v in np.linspace(0, 2*np.pi, 2*n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    for v in np.linspace(0, 2*np.pi, 2*n):
        for u in np.linspace(0, 2*np.pi, n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    points = np.array(points).T
    return R@points, val

def se23_invariant_set_points_v(sol, t, w1_norm, w2_norm, beta): # w1_norm: a, w2_norm: omega
    val = np.real(beta*np.exp(-sol['alpha']*t) + (sol['mu2']*w1_norm**2 + sol['mu3']*w2_norm**2)*(1-np.exp(-sol['alpha']*t)))+0.01 # V(t)
    # 1 = xT(P/V(t))x, equation for the ellipse
    P1 = sol['P']/val
    A1 = P1[0:3, 0:3]
    A2 = P1[0:3, 3:6]
    A3 = P1[0:3, 6:]
    A4 = P1[3:6, :3]
    A5 = P1[3:6, 3:6]
    A6 = P1[3:6, 6:]
    A7 = P1[6:, :3]
    A8 = P1[6:, 3:6]
    A9 = P1[6:, 6:]
    P1 = np.array([[A5, A4, A6],
                   [A2, A1, A3],
                   [A8, A7, A9]]).reshape(9,9)
    A = P1[:3,:3]
    B = P1[:3,3:]
    C = P1[3:,:3]
    D = P1[3:,3:]
    P = A-B@np.linalg.inv(D)@C
    
    evals, evects = np.linalg.eig(P)
    radii = 1/np.sqrt(evals)
    R = evects@np.diag(radii)
    R = np.real(R)
    
    # draw sphere
    points = []
    n = 30
    for u in np.linspace(0, 2*np.pi, n):
        for v in np.linspace(0, 2*np.pi, 2*n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    for v in np.linspace(0, 2*np.pi, 2*n):
        for u in np.linspace(0, 2*np.pi, n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    points = np.array(points).T
    return R@points, val

def se23_invariant_set_points(sol, t, w1_norm, w2_norm, beta): # w1_norm: a, w2_norm: omega
    val = np.real(beta*np.exp(-sol['alpha']*t) + (sol['mu2']*w1_norm**2 + sol['mu3']*w2_norm**2)*(1-np.exp(-sol['alpha']*t))) + 0.1#+0.25 # V(t)
    # 1 = xT(P/V(t))x, equation for the ellipse
    P1 = sol['P']/val
    A1 = P1[:3,:3]
    B1 = P1[:3,3:]
    C1 = P1[3:,:3]
    D1 = P1[3:,3:]
    P = A1-B1@np.linalg.inv(D1)@C1
    
    evals, evects = np.linalg.eig(P)
    radii = 1/np.sqrt(evals)
    R = evects@np.diag(radii)
    R = np.real(R)
    
    # draw sphere
    points = []
    n = 30
    for u in np.linspace(0, 2*np.pi, n):
        for v in np.linspace(0, 2*np.pi, 2*n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    for v in np.linspace(0, 2*np.pi, 2*n):
        for u in np.linspace(0, 2*np.pi, n):
            points.append([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)])
    points = np.array(points).T
    return R@points, val

def exp_map(points, points_theta):
    inv_points = np.zeros((3,points.shape[1]))
    for i in range(points.shape[1]):
        Lie_points = SE3Dcm.wedge(np.array([points[0,i], points[1,i], points[2,i], points_theta[0,i], points_theta[1,i], points_theta[2,i]]))
        exp_points = ca.DM(SE3Dcm.vector(SE3Dcm.exp(Lie_points)))
        exp_points = np.array(exp_points).reshape(6,)
        inv_points[:,i] = np.array([exp_points[0], exp_points[1], exp_points[2]])
    return inv_points

def inv_bound(sol, t, omegabound, abound, ebeta):
    points, val = se23_invariant_set_points(sol, t, omegabound, abound, ebeta)
    points_theta, val = se23_invariant_set_points_theta(sol, t, omegabound, 0.5, ebeta)
    inv_points = np.zeros((3,points.shape[1]))
    for i in range(points.shape[1]):
        Lie_points = SE3Dcm.wedge(np.array([points[0,i], points[1,i], points[2,i], points_theta[0,i], points_theta[1,i], points_theta[2,i]]))
        exp_points = ca.DM(SE3Dcm.vector(SE3Dcm.exp(Lie_points)))
        exp_points = np.array(exp_points).reshape(6,)
        inv_points[:,i] = np.array([exp_points[0], exp_points[1], exp_points[2]])
    xmax = inv_points[0,:].max()
    ymax = inv_points[1,:].max()
    zmax = inv_points[2,:].max()
    xmin = inv_points[0,:].min()
    ymin = inv_points[1,:].min()
    zmin = inv_points[2,:].min()
    return np.array([xmax,ymax,zmax,xmin,ymin,zmin])

def plot_timehis(t_list, T0, dt, PX_list, PY_list, PZ_list, Ppsi_list, sol_LMI, ref, abound, omegabound, n_time, ebeta):
    fig = plt.figure(figsize=(15,15))
    plt.rcParams.update({'font.size': 14})
    fig.subplots_adjust(hspace=0.2, top=0.95)

    # calculte bound along time (small disturbance case)
    T_opt = ref['T0']
    T = np.cumsum(T_opt)
    xr = ref['anchor_x']
    yr = ref['anchor_y']
    zr = ref['anchor_z']

    t_vect = np.linspace(1e-5,T[-1],n_time)
    invbound = np.zeros((6,n_time))
    ref_points = np.zeros((1,n_time))
    for j in range(len(t_vect)):
        for i in range(T.shape[0]):
            if i==0 and t_vect[j] <= T[i]:
                traj_x = np.array(bezier7['bezier7_traj'](t_vect[j], ref['T0'][i], xr[i])).T
                traj_y = np.array(bezier7['bezier7_traj'](t_vect[j], ref['T0'][i], yr[i])).T
                traj_z = np.array(bezier7['bezier7_traj'](t_vect[j], ref['T0'][i], zr[i])).T
                break
            elif T[i-1] < t_vect[j] <= T[i]:
                traj_x = np.array(bezier7['bezier7_traj'](t_vect[j]-np.sum(T_opt[:i]), ref['T0'][i], xr[i])).T
                traj_y = np.array(bezier7['bezier7_traj'](t_vect[j]-np.sum(T_opt[:i]), ref['T0'][i], yr[i])).T
                traj_z = np.array(bezier7['bezier7_traj'](t_vect[j]-np.sum(T_opt[:i]), ref['T0'][i], zr[i])).T

        # reference input at time t
        # world frame
        rx = traj_x[:,0][0]
        ry = traj_y[:,0][0]
        rz = traj_z[:,0][0]
        ib = inv_bound(sol_LMI, 20, abound, omegabound, ebeta)
        ib[0] = rx + ib[0]
        ib[1] = ry + ib[1]
        ib[2] = rz + ib[2]
        ib[3] = rx + ib[3]
        ib[4] = ry + ib[4]
        ib[5] = rz + ib[5]
        invbound[:,j] = ib
        ref_points[:,j] = rz

    freq = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    label_added =False
    for f in freq:
        res = simulate_multirotor(t_list, T0, dt, PX_list, PY_list, PZ_list, Ppsi_list, f)
        
        if not label_added:
            plt.plot(t_list, np.array(res)[:, 2], 'g', label='sim traj',linewidth=0.5)
            label_added = True
        else:
            plt.plot(t_list, np.array(res)[:, 2], 'g',linewidth=0.5)

    t_vect = np.linspace(1e-5,np.cumsum(T_opt)[-1],n_time)
    plt.plot(t_vect, ref_points.reshape(n_time,), 'r', label='ref traj')
    plt.plot(t_vect, invbound[2,:], 'c', label='Bound')
    plt.plot(t_vect, invbound[5,:], 'c')
    plt.xlabel('t, sec')
    plt.ylabel('z, m')
    plt.ylim((-1.5,3.5))
    plt.grid(True)
    plt.legend(loc=2)