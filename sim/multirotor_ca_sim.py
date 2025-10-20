import numpy as np
import matplotlib.pyplot as plt

from cyecca.models.bezier import derive_ref, derive_eulerB321_to_quat, derive_multirotor
from cyecca.lie import SO3EulerB321, SO3Quat
import casadi as ca
from cyecca.models.rdd2 import (
    derive_position_control,
    derive_attitude_control,
    derive_attitude_rate_control,
)
from cyecca.models.rdd2_loglinear import (
    derive_se23_error,
    derive_outerloop_control,
    derive_so3_attitude_control,
)
from cyecca.models.rdd2 import derive_control_allocation
from cyecca.models.quadrotor import derive_model
import pandas as pd

f_alloc = derive_control_allocation()["f_alloc"]
f_euler_to_quat = derive_eulerB321_to_quat()["eulerB321_to_quat"]
f_att_rate_control = derive_attitude_rate_control()["attitude_rate_control"]
f_multirotor = derive_multirotor()["bezier_multirotor"]
f_se23_error = derive_se23_error()["se23_error"]
f_se23_control = derive_outerloop_control()["se23_control"]
f_so3_control = derive_so3_attitude_control()["so3_attitude_control"]
f_se23_att_control = derive_outerloop_control()["se23_attitude_control"]
f_ref = derive_ref()["f_ref"]
model = derive_model()

def simulate_multirotor(t_list, T0, dt, PX_list, PY_list, PZ_list, Ppsi_list, f):
    # loglinear
    model = derive_model()
    x0_dict = model["x0_defaults"]
    x0 = None
    p = None
    if x0 is not None:
        for k in x0.keys():
            if not k in x0_dict.keys():
                raise KeyError(k)
            x0_dict[k] = x0[k]
    p_dict = model["p_defaults"]
    if p is not None:
        for k in p.keys():
            if not k in p_dict.keys():
                raise KeyError(k)
            p_dict[k] = p[k]
    x = np.array(list(x0_dict.values()), dtype=float)
    p = np.array(list(p_dict.values()), dtype=float)
    u = np.zeros(4, dtype=float)
    t1 = 0
    k_p_att = np.array([5, 5, 2], dtype=float)
    leg = 0
    CT = 8.54858e-06

    thrust_trim = 2 * 9.8
    # attitude rate
    kp = np.array([0.3, 0.3, 0.05], dtype=float)
    ki = np.array([0, 0, 0], dtype=float)
    kd = np.array([0.0, 0.0, 0], dtype=float)
    i0 = 0
    e0 = np.zeros(3, dtype=float)  # error for attitude rate loop
    de0 = np.zeros(3, dtype=float)  # deriv of att error (for lowpass)
    f_cut = 10.0
    i_max = np.array([0, 0, 0], dtype=float)
    dae = model["dae"]
    z_i = 0
    pos_list = []
    u_alloc_list = []
    M_ff_list = []
    thrust_ff_list = []
    x_sp_list = []
    y_sp_list = []
    z_sp_list = []
    v_sp_list = []
    a_sp_list = []
    j_sp_list = []
    s_sp_list = []
    motor_omega_list = []
    t1_list = []
    zeta_list = []
    # freq = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    # for f in freq:
    for i in range(len(t_list)):
        t = t_list[i]
        if t > T0 * (leg + 1):
            leg += 1
        t_bezier = t - T0 * leg - dt / 2
        pw = np.array(
            [
                x[model["x_index"]["position_op_w_0"]],
                x[model["x_index"]["position_op_w_1"]],
                x[model["x_index"]["position_op_w_2"]],
            ]
        )
        vw = np.array(
            [
                x[model["x_index"]["velocity_w_p_b_0"]],
                x[model["x_index"]["velocity_w_p_b_1"]],
                x[model["x_index"]["velocity_w_p_b_2"]],
            ]
        )
        q = np.array(
            [
                x[model["x_index"]["quaternion_wb_0"]],
                x[model["x_index"]["quaternion_wb_1"]],
                x[model["x_index"]["quaternion_wb_2"]],
                x[model["x_index"]["quaternion_wb_3"]],
            ]
        )
        omega = np.array(
            [
                x[model["x_index"]["omega_wb_b_0"]],
                x[model["x_index"]["omega_wb_b_1"]],
                x[model["x_index"]["omega_wb_b_2"]],
            ]
        )
        [
            x_sp,
            y_sp,
            z_sp,
            psi_sp,
            dpsi_sp,
            ddpsi_sp,
            vw_sp,
            aw_sp,
            jw_sp,
            sw_sp,
        ] = f_multirotor(
            t_bezier, T0, PX_list[leg], PY_list[leg], PZ_list[leg], Ppsi_list[leg]
        )
        [_, q_sp, omega_ff, _, M_ff, thrust_ff] = f_ref(
            psi_sp, dpsi_sp, ddpsi_sp, vw_sp, aw_sp, jw_sp, sw_sp
        )
        qc_sp = f_euler_to_quat(psi_sp, 0, 0)
        pw_sp = np.array([x_sp, y_sp, z_sp]).reshape(-1)
        zeta = f_se23_error(
            pw,
            vw,
            q,
            pw_sp,
            vw_sp,
            q_sp,
        )
        # position control: world frame
        [thrust, z_i, omega_fb, q_sp] = f_se23_control(
            thrust_trim,
            k_p_att,
            zeta,
            aw_sp,
            q,
            z_i,
            dt,
            t_bezier,
            f
        )
        omega_fb = f_so3_control(np.array([-6.8016, -6.8016, -2.82843], dtype=float), q, q_sp, t_bezier, f)
        # omega_fb = f_se23_att_control(k_p_att, zeta)
        omega_sp = omega_fb + omega_ff
        M_fb, i1, e1, de1, alpha = f_att_rate_control(
            kp,
            ki,
            kd,
            f_cut,
            i_max,
            omega,
            omega_sp,
            i0,
            e0,
            de0,
            dt,
        )
        i0 = i1
        e0 = e1
        de0 = de1
        M_sp = M_fb + M_ff
        u, Fp, Fm, Ft, Msat = f_alloc(100, 0.25/np.sqrt(2), 0.016, CT, thrust, M_sp)
        try:
            f_int = ca.integrator("test", "cvodes", dae, t, t + dt)
            res = f_int(x0=x, z0=0, p=p, u=u)
        except RuntimeError as e:
            print(e)
            xdot = model["f"](x=x, u=u, p=p)
            print(xdot, x, u, p)
            raise e
        t1 += dt
        x = np.array(res["xf"]).reshape(-1)
        pos = x[0:3]
        pos_list.append(np.array(pos).reshape(-1))
        u_alloc_list.append(np.array(u).reshape(-1))
        M_ff_list.append(np.array(M_ff).reshape(-1))
        thrust_ff_list.append(np.array(thrust_ff).reshape(-1))
        zeta_list.append(np.array(zeta).reshape(-1))

    for k in ["xf", "zf"]:
        res[k] = np.array(res[k])
    
    return pos_list

def plot_sim(ref, t_list, T0, PX_list, PY_list, PZ_list, Ppsi_list, flowpipes, num_pipes, axis):

    freq = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    fig = plt.figure(figsize=(15,15))
    plt.rcParams.update({'font.size': 18})
    fig.subplots_adjust(hspace=0.2, top=0.95)

    label_added =False
    for f in freq:
        pos = simulate_multirotor(t_list, T0, 1e-2, PX_list, PY_list, PZ_list, Ppsi_list, f)
        if axis == 'xy':
            if not label_added:
                plt.plot(np.array(pos)[:, 0], np.array(pos)[:, 1], 'g', label='sim traj',linewidth=0.7)
                label_added = True
            else:
                plt.plot(np.array(pos)[:, 0], np.array(pos)[:, 1], 'g',linewidth=0.7)
        elif axis == 'xz':
            if not label_added:
                plt.plot(np.array(pos)[:, 0], np.array(pos)[:, 2], 'g', label='sim traj',linewidth=0.7)
                label_added = True
            else:
                plt.plot(np.array(pos)[:, 0], np.array(pos)[:, 2], 'g',linewidth=0.7)
    if axis == 'xy':
        plt.legend()
        plt.plot(ref['x'], ref['y'], 'r-', label='ref traj')
        # plt.axis('equal')
        plt.grid()
        plt.ylabel('y, m')
        plt.xlim((-2, 7))
        plt.ylim((-7, 7))
    elif axis == 'xz':
        plt.plot(ref['x'], ref['z'], 'r-', label='ref traj')
        plt.ylabel('z, m')
    plt.xlabel('x, m')
    plt.grid(True)

    


    # h_nom = plt.plot(nom[:,0], nom[:,1], color='k', linestyle='-')
    for facet in range(num_pipes):
        hs_ch_LMI = plt.plot(flowpipes[facet][:,0], flowpipes[facet][:,1], color='c', linestyle='--')

    # plt.axis('equal')
    plt.title('Flow Pipes')
    # plt.xlabel('x')
    # plt.ylabel('z')
    lgd = plt.legend(loc=2, prop={'size': 18})
    ax = lgd.axes
    handles, labels = ax.get_legend_handles_labels()
    handles.append(hs_ch_LMI[0])
    labels.append('Flow Pipes')
    lgd._legend_box = None
    lgd._init_legend_box(handles, labels)
    lgd._set_loc(lgd._loc)
    lgd.set_title(lgd.get_title().get_text())

    return 