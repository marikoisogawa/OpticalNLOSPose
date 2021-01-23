from utils import *
from utils.transformation import euler_from_quaternion


def get_global_pos_and_headings(poses, ref=None, sync_intval=None):
    rel_heading = None
    start_pos = None
    ref_pos = None
    pos = []
    headings = []
    for i, pose in enumerate(poses):
        if sync_intval is not None:
            if i % sync_intval == 0:
                rel_heading = quaternion_multiply(get_heading_q(ref[i, 3:7]), quaternion_inverse(get_heading_q(pose[3:7])))
                ref_pos = ref[i, :3]
                start_pos = np.concatenate((pose[:2], ref[i, [2]]))
            p = quat_mul_vec(rel_heading, poses[i, :3] - start_pos)[:2] + ref_pos[:2]
            h = get_heading(quaternion_multiply(rel_heading, get_heading_q(poses[i, 3:7])))
        else:
            p = poses[i, :2]
            h = get_heading(pose[3:7])
        pos.append(p)
        headings.append(h)
    pos = np.vstack(pos)
    headings = np.vstack(headings)
    return pos, headings


def get_joint_angles(poses):
    root_angs = []
    for pose in poses:
        root_euler = np.array(euler_from_quaternion(pose[3:7]))
        root_euler[2] = 0.0
        root_angs.append(root_euler)
    root_angs = np.vstack(root_angs)
    angles = np.hstack((root_angs, poses[:, 7:]))
    return angles


def get_joint_vels(poses, dt):
    vels = []
    for i in range(poses.shape[0] - 1):
        v = get_qvel_fd(poses[i], poses[i+1], dt, 'heading')
        vels.append(v)
    vels = np.vstack(vels)
    return vels


def get_joint_accels(vels, dt):
    accels = np.diff(vels, axis=0) / dt
    accels = np.vstack(accels)
    return accels


def get_mean_dist(x, y):
    return np.linalg.norm(x - y, axis=1).mean()


def get_dist_all(x, y):
    return np.linalg.norm(x - y, axis=0)


def get_mean_abs(x):
    return np.abs(x).mean()


def pad_global_pos(res, traj_orig=None):
    if traj_orig is not None:
        res['traj_orig'] = traj_orig
    traj_pred = res['traj_pred']
    for take in traj_pred.keys():
        traj = traj_pred[take]
        if traj_orig is not None:
            traj_gt = traj_orig[take]
            g_pos = np.tile(traj_gt[0, :2], (traj.shape[0], 1))
        else:
            g_pos = np.tile(np.zeros(2), (traj.shape[0], 1))
        traj_pred[take] = np.hstack([g_pos, traj])

