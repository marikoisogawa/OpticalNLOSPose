import argparse
import os
import sys
import pickle
import math
import time
import glob
import cv2
import yaml
import numpy as np

sys.path.append(os.getcwd())

from ego_pose.utils.metrics import *
from ego_pose.utils.pose_gt import PoseGT
from ego_pose.utils.pose2d import Pose2DContext
from ego_pose.utils.pose_visualization import pose_visualization
from envs.visual.humanoid_vis import HumanoidVisEnv
from ego_pose.utils.statereg_config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--vis-model', default='humanoid_1205_vis_single_v1')
parser.add_argument('--multi-vis-model', default='humanoid_1205_vis_estimate_v1')
parser.add_argument('--multi', action='store_true', default=False)
parser.add_argument('--egomimic-cfg', default=None)
parser.add_argument('--statereg-cfg', default=None)
parser.add_argument('--pathpose-cfg', default=None)
parser.add_argument('--vgail-cfg', default=None)
parser.add_argument('--egomimic-iter', type=int, default=6000)
parser.add_argument('--statereg-iter', type=int, default=100)
parser.add_argument('--vgail-iter', type=int, default=1400)
parser.add_argument('--data', default=None)
parser.add_argument('--meta', default=None)
parser.add_argument('--take-ind', type=int, default=1)
parser.add_argument('--algo-ind', type=int, default=1)
parser.add_argument('--mode', default='vis')
parser.add_argument('--tpv', action='store_true', default=False)
parser.add_argument('--increment', action='store_true', default=False)
# parser.add_argument('--color', action='store_true', default=False)
parser.add_argument('--stats-vis', action='store_true', default=False)
args = parser.parse_args()

dt = 1 / 30.0
data_dir = os.path.expanduser('~/datasets/egopose')
meta = yaml.load(open('%s/meta/%s.yml' % (data_dir, args.meta)))

res_base_dir = os.path.expanduser('~/results/egopose/egopose')
em_res_path = '%s/egomimic/%s/results/iter_%04d_%s.p' % (res_base_dir, args.egomimic_cfg, args.egomimic_iter, args.data)
sr_res_path = '%s/statereg/%s/results/iter_%04d_%s.p' % (res_base_dir, args.statereg_cfg, args.statereg_iter, args.data)
pp_res_path = '%s/pathpose/%s/results/res_meta_%s.p' % (res_base_dir, args.pathpose_cfg, args.meta)
vg_res_path = '%s/vgail/%s/results/iter_%04d_%s.p' % (res_base_dir, args.vgail_cfg, args.vgail_iter, args.data)
em_res, em_meta = pickle.load(open(em_res_path, 'rb')) if args.egomimic_cfg is not None else (None, None)
sr_res, sr_meta = pickle.load(open(sr_res_path, 'rb')) if args.statereg_cfg is not None else (None, None)
pp_res, pp_meta = pickle.load(open(pp_res_path, 'rb')) if args.pathpose_cfg is not None else (None, None)
vg_res, vg_meta = pickle.load(open(vg_res_path, 'rb')) if args.vgail_cfg is not None else (None, None)
if pp_res is not None:
    pad_global_pos(pp_res)
takes = meta['test']
print(takes)

if args.mode == 'stats':

    if args.algo_ind == 0:
        cfg = args.egomimic_cfg
    elif args.algo_ind == 1:
        cfg = args.statereg_cfg

    pose_ctx = Pose2DContext(cfg)

    def eval_take(res, take):
        pose_dist = 0
        traj_pred = res['traj_pred'][take]
        traj_ub = meta['traj_ub'].get(take, traj_pred.shape[0])
        traj_pred = traj_pred[:traj_ub]
        tpv_offset = meta['tpv_offset'].get(take, cfg.fr_margin)
        flip = meta['tpv_flip'].get(take, False)
        fr_num = traj_pred.shape[0]
        valid_num = 0
        for fr in range(max(0, -tpv_offset), fr_num):
            gt_fr = fr + tpv_offset
            gt_file = '%s/gt/poses/%s/%05d_keypoints.json' % (data_dir, take, gt_fr)
            gt_p = pose_ctx.load_gt_pose(gt_file)
            if not pose_ctx.check_gt(gt_p):
                print('invalid frame: %s, %d' % (take, fr))
                continue
            valid_num += 1
            qpos = traj_pred[fr, :]
            p = pose_ctx.align_qpos(qpos, gt_p, flip=flip)
            dist = pose_ctx.get_pose_dist(p, gt_p)
            pose_dist += dist
            if args.stats_vis:
                img = cv2.imread('%s/gt/s_frames/%s/%05d.jpg' % (data_dir, take, gt_fr))
                pose_ctx.draw_pose(img, p * 0.25, flip=flip)
                cv2.imshow('', img)
                cv2.waitKey(1)
        pose_dist /= valid_num
        vels = get_joint_vels(traj_pred, dt)
        accels = get_joint_accels(vels, dt)
        smoothness = get_mean_abs(accels)
        return pose_dist, smoothness


    def compute_metrics(res, algo):
        if res is None:
            return

        print('=' * 10 + ' %s ' % algo + '=' * 10)
        g_pose_dist = 0
        g_smoothness = 0
        for take in takes:
            pose_dist, smoothness = eval_take(res, take)
            g_pose_dist += pose_dist
            g_smoothness += smoothness
            print('%s - pose dist: %.4f, accels: %.4f' % (take, pose_dist, smoothness))
        g_pose_dist /= len(takes)
        g_smoothness /= len(takes)
        print('-' * 60)
        print('all - pose dist: %.4f, accels: %.4f' % (g_pose_dist, g_smoothness))
        print('-' * 60 + '\n')


    compute_metrics(em_res, 'ego mimic')
    compute_metrics(vg_res, 'vgail')
    compute_metrics(pp_res, 'path pose')
    compute_metrics(sr_res, 'state reg')

elif args.mode == 'vis':

    def key_callback(key, action, mods):
        global T, fr, paused, stop, reverse, algo_ind, take_ind, tpv_offset, mfr_int

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
            update_all()
        elif key == glfw.KEY_S:
            reverse = not reverse
        elif key == glfw.KEY_Q:
            tpv_offset -= 1
            update_images()
            print('tpv offset: %d' % tpv_offset)
        elif key == glfw.KEY_E:
            tpv_offset += 1
            update_images()
            print('tpv offset: %d' % tpv_offset)
        elif key == glfw.KEY_Z:
            fr = 0
            take_ind = (take_ind - 1) % len(takes)
            load_take()
            update_all()
        elif key == glfw.KEY_C:
            fr = 0
            take_ind = (take_ind + 1) % len(takes)
            load_take()
            update_all()
        elif key == glfw.KEY_V:
            env_vis.focus = not env_vis.focus
        elif glfw.KEY_1 <= key < glfw.KEY_1 + len(algos):
            algo_ind = key - glfw.KEY_1
            load_take(False)
            update_all()
        elif key == glfw.KEY_MINUS:
            mfr_int -= 1
            update_pose()
        elif key == glfw.KEY_EQUAL:
            mfr_int += 1
            update_pose()
        elif key == glfw.KEY_RIGHT:
            if fr < traj_pred.shape[0] - 1 and fr < len(traj_fpv) - 1:
                fr += 1
            update_all()
        elif key == glfw.KEY_LEFT:
            if fr > 0:
                fr -= 1
            update_all()
        elif key == glfw.KEY_SPACE:
            paused = not paused
        # elif key == glfw.KEY_L:
        #     env_vis.viewer.cam.lookat[0] -= 1
        #     print('env_vis.viewer.cam.lookat[0] offset: %d' % env_vis.viewer.cam.lookat[0])
        # elif key == glfw.KEY_R:
        #     env_vis.viewer.cam.lookat[0] += 1
        #     print('env_vis.viewer.cam.lookat[0] offset: %d' % env_vis.viewer.cam.lookat[0])
        else:
            return False

        return True


    def update_pose():
        global fr
        print('take_ind: %d, fr: %d, tpv fr: %d, mfr int: %d' % (take_ind, fr, fr + tpv_offset, mfr_int))
        if args.multi:
            nq = 59
            num_model = env_vis.model.nq // nq
            vec = np.array([0, -1])
            # env_vis.viewer.cam.lookat[0] = -0.5
            hq = get_heading_q(traj_pred[fr, 3:7])
            rel_q = quaternion_multiply(hq, quaternion_inverse(get_heading_q(traj_pred[fr, 3:7])))
            # vec = quat_mul_vec(hq, np.array([0, -1, 0]))[:2]
            vec = np.array([0, -1])
            env_vis.viewer._hide_overlay = True
            env_vis.viewer.cam.lookat[:2] = traj_pred[fr, :2] + vec * 0.8 * 6
            env_vis.viewer.cam.azimuth = 0.0
            env_vis.viewer.cam.elevation = 0.0
            env_vis.viewer.cam.distance = 10.0
            fr = 1448
            for i in range(num_model):
                fr_m = min(fr + i * mfr_int, traj_pred.shape[0] - 1)
                env_vis.data.qpos[i * nq: (i + 1) * nq] = traj_pred[fr_m, :]
                env_vis.data.qpos[i * nq + 3: i * nq + 7] = de_heading(traj_pred[fr_m, 3:7])
                env_vis.data.qpos[i * nq: i * nq + 2] = traj_pred[fr, :2] + vec * 0.8 * i
        else:
            env_vis.data.qpos[:] = traj_pred[fr, :]
        # env_vis.data.qpos[3:7] = [1, 0, 0, 0]
        # env_vis.viewer.cam.lookat[:2] = traj_pred[fr + 6 * mfr_int, :2]
        env_vis.sim_forward()
        print(env_vis.viewer.cam.lookat)


    # def update_images():
    #     cv2.imshow('depth', traj_fpv[fr])
    #     cv2.imshow('color', traj_color[fr])


    def update_all():
        update_pose()
        # update_images()

    def load_take(load_images=True):
        global traj_pred, traj_fpv, traj_tpv, traj_color, tpv_offset
        algo_res, algo = algos[algo_ind]
        if algo_res is None:
            return
        take = takes[take_ind]
        traj_pred = algo_res['traj_pred'][take]
        print('%s ---------- %s' % (algo, take))


    traj_pred = None
    traj_fpv = None
    traj_color = None
    traj_tpv = None
    vis_model_file = 'assets/mujoco_models/%s.xml' % (args.multi_vis_model if args.multi else args.vis_model)
    env_vis = HumanoidVisEnv(vis_model_file, 1, focus=not args.multi)
    env_vis.set_custom_key_callback(key_callback)
    algos = [(em_res, 'ego mimic'), (sr_res, 'state reg'), (pp_res, 'path pose'), (vg_res, 'vgail')]
    algo_ind = args.algo_ind
    tpv_offset = 0
    mfr_int = 15
    load_take()

    """render or select part of the clip"""
    env_vis.viewer._hide_overlay = True
    T = 10
    paused = False
    stop = False
    reverse = False

    if args.increment:
        for i in range(args.take_ind, len(takes)):
            take_ind = i
            T = 7
            t = 0
            fr = 0
            tpv_offset = 0
            mfr_int = 2
            load_take()
            update_all()
            stop = False
            while not stop:
                if t >= math.floor(T):
                    if not reverse and fr < traj_pred.shape[0] - 1:
                        fr += 1
                        update_all()
                    elif reverse and fr > 0:
                        fr -= 1
                        update_all()
                    t = 0
                env_vis.render()
                if not paused:
                    t += 1
                if fr >= traj_pred.shape[0] - 1:
                    stop = True
                    break
    else:
        take_ind = args.take_ind
        t = 0
        fr = 0
        load_take()
        update_all()
        while not stop:
            if t >= math.floor(T):
                if not reverse and fr < traj_pred.shape[0] - 1:
                    fr += 1
                    update_all()
                elif reverse and fr > 0:
                    fr -= 1
                    update_all()
                t = 0

            env_vis.render()
            if not paused:
                t += 1

