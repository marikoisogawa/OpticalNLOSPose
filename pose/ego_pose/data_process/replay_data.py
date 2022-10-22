# -------------------------------------------------
# Sample command: 
# ego_pose/data_process/replay_data.py --mocap-id 0702 --meta-id meta_0702 --take-ind 0
# 
# How to use:
#  - Please pless Q or E key to find synchronize
#  - It outputs offset parameter. Please use this offset for mocap sync parameter
#  - If the z potision of humanoid foot is weird, please set offset-z with up and down key. You can reflect this z offset value to the mocap data with convert_clip.
# -------------------------------------------------

import os
import sys
import numpy as np
import math
sys.path.append(os.getcwd())

from utils import *
from mujoco_py import load_model_from_path, MjSim
from envs.common.mjviewer import MjViewer
import pickle
import glob
import argparse
import glfw
import yaml
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--model-id', type=str, default='humanoid_1205_vis_single_v1')
parser.add_argument('--mocap-id', type=str, default='1030')
parser.add_argument('--meta-id', type=str, default='meta_1030')
parser.add_argument('--offset-z', type=float, default=0.067)
parser.add_argument('--take-ind', type=int, default=0)
parser.add_argument('--multi', action='store_true', default=False)

args = parser.parse_args()

model_file = 'assets/mujoco_models/%s.xml' % ('humanoid_1205_vis_multi_v1' if args.multi else args.model_id)
model = load_model_from_path(model_file)
sim = MjSim(model)
viewer = MjViewer(sim)

if args.meta_id is not None:
    meta_file = os.path.expanduser('~/datasets/egopose/meta/%s.yml' % args.meta_id)
    meta = yaml.load(open(meta_file, 'r'))
take_folders = glob.glob(os.path.expanduser('~/datasets/egopose/depth/%s_*' % args.mocap_id))
take_folders.sort()
takes = [os.path.splitext(os.path.basename(x))[0] for x in take_folders]


def key_callback(key, action, mods):
    global T, fr, paused, stop, im_offset, offset_z, take_ind, reverse

    if action != glfw.RELEASE:
        return False
    elif key == glfw.KEY_D:
        T *= 1.5
    elif key == glfw.KEY_F:
        T = max(1, T / 1.5)
    elif key == glfw.KEY_R:
        stop = True
    elif key == glfw.KEY_Q:
        if fr + im_offset > 0:
            im_offset -= 1
        update_depth()
    elif key == glfw.KEY_E:
        if fr + im_offset < len(images) - 1:
            im_offset += 1
        update_depth()
    elif key == glfw.KEY_W:
        fr = max(0, -im_offset)
        update_all()
    elif key == glfw.KEY_S:
        reverse = not reverse
    elif key == glfw.KEY_C:
        take_ind = (take_ind + 1) % len(takes)
        load_take()
        update_all()
    elif key == glfw.KEY_Z:
        take_ind = (take_ind - 1) % len(takes)
        load_take()
        update_all()
    elif key == glfw.KEY_RIGHT:
        if fr + im_offset < len(images) - 1 and fr < qpos_traj.shape[0] - 1:
            fr += 1
        update_all()
    elif key == glfw.KEY_LEFT:
        if fr + im_offset > 0 and fr > 0:
            fr -= 1
        update_all()
    elif key == glfw.KEY_UP:
        offset_z += 0.001
        update_mocap()
    elif key == glfw.KEY_DOWN:
        offset_z -= 0.001
        update_mocap()
    elif key == glfw.KEY_SPACE:
        paused = not paused
    else:
        return False
    return True


def update_mocap():
    if args.multi:
        nq = 59
        num_model = sim.model.nq // nq
        hq = get_heading_q(qpos_traj[fr, 3:7])
        vec = quat_mul_vec(hq, np.array([0, -1, 0]))[:2]
        for i in range(num_model):
            fr_m = min(fr + i * 10, qpos_traj.shape[0] - 1)
            sim.data.qpos[i*nq: (i+1)*nq] = qpos_traj[fr_m, :]
            sim.data.qpos[i*nq + 2] += offset_z
            sim.data.qpos[i * nq: i * nq + 2] = sim.data.qpos[:2] + vec * 0.8 * i
    else:
        sim.data.qpos[:] = qpos_traj[fr]
        sim.data.qpos[2] += offset_z
    sim.forward()


def update_depth():
    print('take: %s  fr: %d  im_fr: %d  offset: %d  fr_boundary: (%d, %d)  dz: %.3f' %
          (takes[take_ind], fr, fr + im_offset, im_offset, max(-im_offset, 0),
           min(qpos_traj.shape[0], len(images) - im_offset - 1), offset_z))
    cv2.imshow('depth', images[fr + im_offset])


def update_all():
    update_mocap()
    update_depth()


def load_take():
    global qpos_traj, images, im_offset, fr
    take = takes[take_ind]
    im_offset = 0 if args.meta_id is None else meta['video_mocap_sync'][take][0]
    fr = max(0, -im_offset)
    traj_file = os.path.expanduser('~/datasets/egopose/bvh/%s_traj.p' % take)
    qpos_traj = pickle.load(open(traj_file, "rb"))
    frame_folder = os.path.expanduser('~/datasets/egopose/depth/%s' % take)
    frame_files = glob.glob(os.path.join(frame_folder, '*.png'))
    frame_files.sort()
    images = [cv2.imread(file) for file in frame_files]
    print('traj len: %d,  image num: %d,  dz: %.3f' % (qpos_traj.shape[0], len(images), offset_z))


qpos_traj = None
images = None
take_ind = args.take_ind

T = 10
im_offset = 0
offset_z = args.offset_z
fr = max(0, -im_offset)
paused = False
stop = False
reverse = False

cv2.namedWindow('depth')
cv2.moveWindow('depth', 1450, 0)
glfw.set_window_size(viewer.window, 1000, 960)
glfw.set_window_pos(viewer.window, 2500, 0)
viewer._hide_overlay = True
viewer.cam.distance = 10
viewer.cam.elevation = -20
viewer.cam.azimuth = 90
viewer.custom_key_callback = key_callback

load_take()
update_depth()
update_mocap()
t = 0
while not stop:
    if t >= math.floor(T):
        if not reverse and fr + im_offset < len(images) - 1 and fr < qpos_traj.shape[0] - 1:
            fr += 1
            update_all()
        elif reverse and fr + im_offset > 0 and fr > 0:
            fr -= 1
            update_all()
        t = 0

    viewer.render()
    if not paused:
        t += 1