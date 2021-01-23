import json
import numpy as np
import math
import cv2
from ego_pose.envs.humanoid_v1 import HumanoidEnv
from envs.visual.humanoid_vis import HumanoidVisEnv


class Pose2DContext:

    def __init__(self, cfg, gt_comparison=False):
        if gt_comparison:
            self.vis_model = 'humanoid_1205_vis_single_v1'
            self.vis_model_file = 'assets/mujoco_models/%s.xml' % self.vis_model
            self.multi = False
            self.env = HumanoidVisEnv(self.vis_model_file, 1, focus=not self.multi)
        else:
            self.env = HumanoidEnv(cfg)
        self.body2id = self.env.model._body_name2id
        self.body_names = self.env.model.body_names[1:]
        self.body_set = {'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand', 'LeftArm', 'RightArm',
                         'LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg', 'LeftFoot', 'RightFoot'}
        self.nbody = len(self.body_set)
        self.body_filter = np.zeros((len(self.body_names),), dtype=bool)
        for body in self.body2id.keys():
            if body in self.body_set:
                self.body_filter[self.body2id[body]-1] = True
        self.body_names = [self.body_names[i] for i in range(len(self.body_filter)) if self.body_filter[i]]
        self.body2id = {body: i for i, body in enumerate(self.body_names)}

        self.conn = [('RightUpLeg', 'RightArm', (255, 255, 0)),
                     ('RightArm', 'RightForeArm', (255, 191, 0)),
                     ('RightForeArm', 'RightHand', (255, 191, 0)),
                     ('RightUpLeg', 'RightLeg', (255, 64, 0.0)),
                     ('RightLeg', 'RightFoot', (255, 64, 0.0)),
                     ('LeftUpLeg', 'LeftArm', (0, 255, 128)),
                     ('LeftArm', 'LeftForeArm', (0, 255, 255)),
                     ('LeftForeArm', 'LeftHand', (0, 255, 255)),
                     ('LeftUpLeg', 'LeftLeg', (0, 64, 255)),
                     ('LeftLeg', 'LeftFoot', (0, 64, 255))]

        self.joints_map = [(2, self.body2id['RightArm']),
                           (3, self.body2id['RightForeArm']),
                           (4, self.body2id['RightHand']),
                           (5, self.body2id['LeftArm']),
                           (6, self.body2id['LeftForeArm']),
                           (7, self.body2id['LeftHand']),
                           (9, self.body2id['RightUpLeg']),
                           (10, self.body2id['RightLeg']),
                           (11, self.body2id['RightFoot']),
                           (12, self.body2id['LeftUpLeg']),
                           (13, self.body2id['LeftLeg']),
                           (14, self.body2id['LeftFoot'])]

        """
        Since number of joints between
         - human_dynamics (25 joints)
         - ours (14 joints)
        are different, the following mapping variable defines
        which joint to be evaluated.
        
        0: indices for human_dynamics
        1: indices for ours
        The name after # shows corresponded name of human_dynamics
        """
        self.joints_map1 = [(8,  self.body2id['RightArm']),      # Right shoulder
                            (7,  self.body2id['RightForeArm']),  # Right elbow
                            (6,  self.body2id['RightHand']),     # Right wrist
                            (9,  self.body2id['LeftArm']),       # Left shoulder
                            (10, self.body2id['LeftForeArm']),   # Left elbow
                            (11, self.body2id['LeftHand']),      # Left wrist
                            (2,  self.body2id['RightUpLeg']),    # Right hip
                            (1,  self.body2id['RightLeg']),      # Right knee
                            (0,  self.body2id['RightFoot']),     # Right heel
                            (3,  self.body2id['LeftUpLeg']),     # Left hip
                            (4,  self.body2id['LeftLeg']),       # Left knee
                            (5,  self.body2id['LeftFoot'])]      # Left heel

    def draw_pose(self, img, pose, flip=False):
        conn = self.conn
        if flip:
            conn = self.conn[5:] + self.conn[:5]
        for b1, b2, c in conn:
            p1 = pose[self.body2id[b1], :2]
            p2 = pose[self.body2id[b2], :2]
            self.draw_bone(img, p1, p2, c)
        for x in self.body_set:
            e = pose[self.body2id[x], :2]
            cv2.circle(img, (int(e[0]), int(e[1])), 1, (0, 0, 255), -1)

    def draw_bone(self, img, p1, p2, c):
        center = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))
        angle = int(math.atan2(p2[1]-p1[1], p2[0]-p1[0]) / np.pi * 180)
        axes = (int(np.linalg.norm(p2-p1)/2), 1)
        cv2.ellipse(img, center, axes, angle, 0, 360, c, -1)

    def load_gt_pose(self, filename):
        data = json.load(open(filename))
        keypoints = data['people'][0]['pose_keypoints_2d']
        p = np.zeros((self.nbody, 3))
        for i1, i2 in self.joints_map:
            p[i2, :] = keypoints[3*i1: 3*i1 + 3]
        return p

    def check_gt(self, gt_pose):
        return gt_pose[self.body2id['LeftUpLeg'], 2] > 0.1 or gt_pose[self.body2id['RightUpLeg'], 2] > 0.1

    def get_pose_dist(self, p, gt_p):
        body2id = self.body2id
        if gt_p[body2id['LeftArm'], 2] > 0.1 and gt_p[body2id['LeftUpLeg'], 2] > 0.1:
            kp1 = 'LeftArm'
            kp2 = 'LeftUpLeg'
        else:
            kp1 = 'RightArm'
            kp2 = 'RightUpLeg'
        scale = 0.5 / abs(gt_p[body2id[kp1], 1] - gt_p[body2id[kp2], 1])

        dist = 0
        num = 0
        for i in range(gt_p.shape[0]):
            if gt_p[i, 2] > 0.1:
                dist += np.linalg.norm(gt_p[i, :2] - p[i, :]) * scale
                num += 1
        dist /= num
        return dist

    def project_qpos(self, qpos, flip):
        self.env.data.qpos[:] = qpos
        self.env.sim.forward()
        pose_3d = np.vstack(self.env.data.body_xpos[1:])
        pose_3d = pose_3d[self.body_filter, :]
        body2id = self.body2id

        """make projection matrix"""
        vp = (pose_3d[body2id['LeftUpLeg'], :] + pose_3d[body2id['RightUpLeg'], :]) * 0.5
        v = pose_3d[body2id['RightUpLeg'], :] - pose_3d[body2id['LeftUpLeg'], :]
        if flip:
            v *= -1
        v[2] = 0
        v /= np.linalg.norm(v)
        x = v
        z = np.array([0, 0, 1])
        y = np.cross(z, x)
        # R, t transfrom camera coordinate to world coordiante
        R = np.hstack((-y[:, None], z[:, None], x[:, None]))
        t = vp - 10 * x
        t = t[:, None]
        E = np.hstack((R.T, -R.T.dot(t)))

        p = np.hstack((pose_3d, np.ones((pose_3d.shape[0], 1)))).dot(E.T)
        p = p[:, :2] / p[:, [2]]
        p[:, 1] *= -1
        return p

    def align_qpos(self, qpos, gt_p, scale=None, flip=False):
        body2id = self.body2id
        p = self.project_qpos(qpos, flip)
        base = np.zeros((1, 2))
        n = 0
        if gt_p[body2id['LeftUpLeg'], 2] > 0.1:
            base += gt_p[[body2id['LeftUpLeg']], :2]
            n += 1
        if gt_p[body2id['RightUpLeg'], 2] > 0.1:
            base += gt_p[[body2id['RightUpLeg']], :2]
            n += 1
        base /= n

        if scale is None:
            if gt_p[body2id['LeftLeg'], 2] > 0.1 and gt_p[body2id['LeftUpLeg'], 2] > 0.1:
                kp1 = 'LeftLeg'
                kp2 = 'LeftUpLeg'
            else:
                kp1 = 'RightLeg'
                kp2 = 'RightUpLeg'
            scale = np.linalg.norm(gt_p[body2id[kp1]] - gt_p[body2id[kp2]]) / np.linalg.norm(p[body2id[kp1]] - p[body2id[kp2]])

        p = p * scale + base
        return p

    def project_qpos_for_gt_comparison(self, qpos, flip):
        self.env.data.qpos[:] = qpos
        self.env.sim.forward()
        pose_3d = np.vstack(self.env.data.body_xpos[1:])
        pose_3d = pose_3d[self.body_filter, :]
        body2id = self.body2id

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], c='r', marker='o')
        # plt.scatter(pose_3d[:, 0], pose_3d[:, 2])
        # plt.xlim([-0.5, 0.5])
        # plt.ylim([0, 1.3])
        # plt.pause(0.001)
        # plt.clf()
        return pose_3d

    def align_qpos_for_gt_comparison(self, qpos, scale=None, flip=False):
        body2id = self.body2id
        p = self.project_qpos_for_gt_comparison(qpos, flip)

        """ Average L2 norm between Right/Left UpLeg and Arm """
        scale = (np.linalg.norm(p[body2id['RightUpLeg']] - p[body2id['RightArm']], ord=2)
                 + np.linalg.norm(p[body2id['LeftUpLeg']] - p[body2id['LeftArm']], ord=2)) / 2.0
        scale = 0.5 / scale

        """ Root Position Offset """
        root_pos = (p[body2id['RightUpLeg']] + p[body2id['LeftUpLeg']]) / 2.0

        """ Offset to Move Joints (= Origin Position) """
        offset = [0, 0, 0]

        """ Adjusted 3D Pose """
        # pose_3d = (p - root_pos) * scale + offset
        pose_3d = p - root_pos

        """ Swap 1 and 2 channels to make (x,y,z) alignment """
        tmp = pose_3d[:, 1].copy()
        pose_3d[:, 1] = pose_3d[:, 2]
        pose_3d[:, 2] = tmp

        """ Compute 2D Pose """
        pose_2d = np.zeros((pose_3d.shape[0], 2))
        pose_2d[:, 0:2] = pose_3d[:, 0:2]

        return pose_2d, pose_3d

    """
    Compute pose distance for all joints
    """
    def get_pose_dist_all(self, p_pred, p_gt):
        dist = 0
        for i in range(len(self.joints_map1)):
            dist += np.linalg.norm(p_pred[self.joints_map1[i][1], :] - p_gt[self.joints_map1[i][0], :], ord=2)
        dist /= len(self.joints_map1)
        return dist

    """
    Compute pose distance for end effector joints (hand, foot) only
    """
    def get_pose_dist_ee(self, p_pred, p_gt):
        dist = 0
        for i in range(len(self.joints_map1)):
            if self.joints_map1[i][1] == self.body2id['RightHand']:
                dist += np.linalg.norm(p_pred[self.joints_map1[i][1], :] - p_gt[self.joints_map1[i][0], :], ord=2)
            elif self.joints_map1[i][1] == self.body2id['LeftHand']:
                dist += np.linalg.norm(p_pred[self.joints_map1[i][1], :] - p_gt[self.joints_map1[i][0], :], ord=2)
            elif self.joints_map1[i][1] == self.body2id['RightFoot']:
                dist += np.linalg.norm(p_pred[self.joints_map1[i][1], :] - p_gt[self.joints_map1[i][0], :], ord=2)
            elif self.joints_map1[i][1] == self.body2id['LeftFoot']:
                dist += np.linalg.norm(p_pred[self.joints_map1[i][1], :] - p_gt[self.joints_map1[i][0], :], ord=2)
        dist /= 4.0
        return dist

    # """
    # Return pose for end effector joints (hand, foot) only
    # """
    # def get_pose_ee(self, p_pred):
    #     p_pred_ee = np.zeros((p_pred.shape[0], 4))
    #     for i in range(len(self.joints_map1)):
    #         if self.joints_map1[i][1] == self.body2id['RightHand']:
    #             p_pred_ee[:, 0] = p_pred[:, self.joints_map1[i][1]]
    #         elif self.joints_map1[i][1] == self.body2id['LeftHand']:
    #             p_pred_ee[:, 1] = p_pred[:, self.joints_map1[i][1]]
    #         elif self.joints_map1[i][1] == self.body2id['RightFoot']:
    #             p_pred_ee[:, 2] = p_pred[:, self.joints_map1[i][1]]
    #         elif self.joints_map1[i][1] == self.body2id['LeftFoot']:
    #             p_pred_ee[:, 3] = p_pred[:, self.joints_map1[i][1]]
    #     return p_pred_ee

    """
    Get pose distance between GT and estimated
    """
    def get_pose_dist_for_gt_comparison(self, p_pred, p_gt, mode):
        if mode == 'all_joints':
            return self.get_pose_dist_all(p_pred, p_gt)
        elif mode == 'end_effector':
            return self.get_pose_dist_ee(p_pred, p_gt)

