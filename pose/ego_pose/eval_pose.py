import argparse
import os
import sys
import pickle
import math
import time
import numpy as np
import yaml
sys.path.append(os.getcwd())

from ego_pose.utils.metrics import *
from ego_pose.envs.humanoid_v1 import HumanoidEnv
from envs.visual.humanoid_vis import HumanoidVisEnv


parser = argparse.ArgumentParser()
parser.add_argument('--vis-model', default='humanoid_1205_vis_double_v1')
parser.add_argument('--multi-vis-model', default='humanoid_1205_vis_estimate_v1')
parser.add_argument('--multi', action='store_true', default=False)
parser.add_argument('--meta', default='meta_s1_06')
parser.add_argument('--egomimic-cfg', default=None)
parser.add_argument('--statereg-cfg', default=None)
parser.add_argument('--pathpose-cfg', default=None)
parser.add_argument('--vgail-cfg', default=None)
parser.add_argument('--egomimic-iter', type=int, default=6000)
parser.add_argument('--statereg-iter', type=int, default=100)
parser.add_argument('--vgail-iter', type=int, default=1400)
parser.add_argument('--algo-ind', type=int, default=1)
parser.add_argument('--egomimic-tag', default='')
parser.add_argument('--data', default='train')
parser.add_argument('--mode', default='stats')  # vis or stats
parser.add_argument('--take_ind', type=int, default=0)
args = parser.parse_args()


def compute_metrics(results, meta, algo):
    if results is None:
        return

    print('=' * 10 + ' f%s ' % algo + '=' * 10)
    g_pose_dist = 0
    g_vel_dist = 0
    g_smoothness = 0
    g_global_pos_dist = 0
    g_heading_diff = 0
    traj_orig = results['traj_orig']
    traj_pred = results['traj_pred']

    for take in traj_pred.keys():
        traj = traj_pred[take]
        traj_gt = traj_orig[take]
        traj[:, 32:35] = 0.0
        traj[:, 42:45] = 0.0
        traj_gt[:, 32:35] = 0.0
        traj_gt[:, 42:45] = 0.0
        # compute gt stats
        angs_gt = get_joint_angles(traj_gt)
        vels_gt = get_joint_vels(traj_gt, dt)
        pos_gt, headings_gt = get_global_pos_and_headings(traj_gt)
        # compute pred stats
        angs = get_joint_angles(traj)
        vels = get_joint_vels(traj, dt)
        accels = get_joint_accels(vels, dt)
        pos, headings = get_global_pos_and_headings(traj, traj_gt, 60)
        # compute metrics
        pose_dist = get_mean_dist(angs, angs_gt)
        vel_dist = get_mean_dist(vels, vels_gt)
        smoothness = get_mean_abs(accels)
        global_pos_dist = get_mean_dist(pos, pos_gt)
        heading_diff = get_mean_abs(headings - headings_gt)
        g_pose_dist += pose_dist
        g_vel_dist += vel_dist
        g_smoothness += smoothness
        g_global_pos_dist += global_pos_dist
        g_heading_diff += heading_diff

        print('%s - pose dist: %.4f, vel dist: %.4f, accels: %.4f, global pos dist: %.4f, heading diff: %.4f' %
              (take, pose_dist, vel_dist, smoothness, global_pos_dist, heading_diff))

    g_pose_dist /= len(traj_pred)
    g_vel_dist /= len(traj_pred)
    g_smoothness /= len(traj_pred)
    g_global_pos_dist /= len(traj_pred)
    g_heading_diff /= len(traj_pred)

    print('-' * 60)
    print('all - pose dist: %.4f, vel dist: %.4f, accels: %.4f, global pos dist: %.4f, heading diff: %.4f, num reset: %d' %
          (g_pose_dist, g_vel_dist, g_smoothness, g_global_pos_dist, g_heading_diff, meta.get('num_reset', 0)))
    print('-' * 60 + '\n')


dt = 1 / 30.0
data_dir = os.path.expanduser('~/datasets/egopose')
meta = yaml.load(open('%s/meta/%s.yml' % (data_dir, args.meta)))

res_base_dir = os.path.expanduser('~/results/egopose/egopose')
em_res_path = '%s/egomimic/%s/results/iter_%04d_%s%s.p' % (res_base_dir, args.egomimic_cfg, args.egomimic_iter, args.data, args.egomimic_tag)
sr_res_path = '%s/statereg/%s/results/iter_%04d_%s.p' % (res_base_dir, args.statereg_cfg, args.statereg_iter, args.data)
pp_res_path = '%s/pathpose/%s/results/res_%s.p' % (res_base_dir, args.pathpose_cfg, args.data)
vg_res_path = '%s/vgail/%s/results/iter_%04d_%s.p' % (res_base_dir, args.vgail_cfg, args.vgail_iter, args.data)
em_res, em_meta = pickle.load(open(em_res_path, 'rb')) if args.egomimic_cfg is not None else (None, None)
sr_res, sr_meta = pickle.load(open(sr_res_path, 'rb')) if args.statereg_cfg is not None else (None, None)
pp_res, pp_meta = pickle.load(open(pp_res_path, 'rb')) if args.pathpose_cfg is not None else (None, None)
vg_res, vg_meta = pickle.load(open(vg_res_path, 'rb')) if args.vgail_cfg is not None else (None, None)
if pp_res is not None:
    pad_global_pos(pp_res, em_res['traj_orig'])

if args.mode == 'stats':
    if args.egomimic_cfg is not None:
        compute_metrics(em_res, em_meta, 'ego mimic')
    if args.statereg_cfg is not None:
        compute_metrics(sr_res, sr_meta, 'state reg')

elif args.mode == 'vis':
    """visualization"""

    def key_callback(key, action, mods):
        global T, fr, paused, stop, reverse, algo_ind, take_ind, ss_ind, show_gt, mfr_int

        if action != glfw.RELEASE:
            return False
        if key == glfw.KEY_D:
            T *= 1.5
        elif key == glfw.KEY_F:
            T = max(1, T / 1.5)
        elif key == glfw.KEY_R:
            stop = True
        elif key == glfw.KEY_W:
            fr = 0
            update_pose()
        elif key == glfw.KEY_S:
            reverse = not reverse
        elif key == glfw.KEY_Q:
            fr = 0
            take_ind = (take_ind - 1) % len(takes)
            load_take()
            update_pose()
        elif key == glfw.KEY_E:
            fr = 0
            take_ind = (take_ind + 1) % len(takes)
            load_take()
            update_pose()
        elif key == glfw.KEY_X:
            save_screen_shots(env_vis.viewer.window, 'out/%04d.png' % ss_ind)
            ss_ind += 1
        elif glfw.KEY_1 <= key < glfw.KEY_1 + len(algos):
            algo_ind = key - glfw.KEY_1
            load_take()
            update_pose()
        elif key == glfw.KEY_0:
            show_gt = not show_gt
            update_pose()
        elif key == glfw.KEY_MINUS:
            mfr_int -= 1
            update_pose()
        elif key == glfw.KEY_EQUAL:
            mfr_int += 1
            update_pose()
        elif key == glfw.KEY_RIGHT:
            if fr < traj_orig.shape[0] - 1:
                fr += 1
            update_pose()
        elif key == glfw.KEY_LEFT:
            if fr > 0:
                fr -= 1
            update_pose()
        elif key == glfw.KEY_SPACE:
            paused = not paused
        else:
            return False

        return True


    def de_heading(q):
        return quaternion_multiply(quaternion_inverse(get_heading_q(q)), q)

    def update_pose():
        global fr
        print('take_ind: %d, fr: %d, mfr int: %d' % (take_ind, fr, mfr_int))
        if args.multi:
            nq = 59
            # traj = traj_orig if show_gt else traj_pred
            traj = traj_orig
            num_model = env_vis.model.nq // nq
            hq = get_heading_q(traj_orig[fr, 3:7])
            rel_q = quaternion_multiply(hq, quaternion_inverse(get_heading_q(traj[fr, 3:7])))
            # vec = quat_mul_vec(hq, np.array([0, -1, 0]))[:2]
            vec = np.array([0, -1])
            env_vis.viewer._hide_overlay = True
            env_vis.viewer.cam.lookat[:2] = traj_orig[fr, :2] + vec * 0.8 * 6
            env_vis.viewer.cam.azimuth = 0.0
            env_vis.viewer.cam.elevation = 0.0
            env_vis.viewer.cam.distance = 10.0
            fr = 207
            for i in range(num_model):
                fr_m = min(fr + i * mfr_int, traj_orig.shape[0] - 1)
                env_vis.data.qpos[i * nq: (i + 1) * nq] = traj_orig[fr_m, :]
                env_vis.data.qpos[i * nq + 3: i * nq + 7] = de_heading(traj_orig[fr_m, 3:7])
                env_vis.data.qpos[i * nq: i * nq + 2] = traj_orig[fr, :2] + vec * 0.8 * i
        else:
            nq = env_vis.model.nq // 2
            env_vis.data.qpos[:nq] = traj_orig[fr, :]
            # add x offset
            env_vis.data.qpos[nq] += 1.0
        env_vis.sim_forward()

    def load_take():
        global traj_pred, traj_orig
        algo_res, algo = algos[algo_ind]
        if algo_res is None:
            return
        take = takes[take_ind]
        print('%s ---------- %s' % (algo, take))
        traj_pred = algo_res['traj_pred'][take]
        traj_orig = algo_res['traj_orig'][take]

    traj_pred = None
    traj_orig = None
    vis_model_file = 'assets/mujoco_models/%s.xml' % (args.multi_vis_model if args.multi else args.vis_model)
    env_vis = HumanoidVisEnv(vis_model_file, 1, focus=not args.multi)
    env_vis.set_custom_key_callback(key_callback)
    takes = meta[args.data]
    algos = [(em_res, 'ego mimic'), (sr_res, 'state reg'), (pp_res, 'path pose'), (vg_res, 'vgail')]
    algo_ind = args.algo_ind
    take_ind = args.take_ind
    ss_ind = 0
    mfr_int = 14
    show_gt = False
    load_take()

    """render or select part of the clip"""
    T = 10
    fr = 0
    paused = False
    stop = False
    reverse = False

    update_pose()
    t = 0
    while not stop:
        if t >= math.floor(T):
            if not reverse and fr < traj_orig.shape[0] - 1:
                fr += 1
                update_pose()
            elif reverse and fr > 0:
                fr -= 1
                update_pose()
            t = 0

        env_vis.render()
        if not paused:
            t += 1

