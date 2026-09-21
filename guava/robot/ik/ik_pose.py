import numpy as np
import os


def _quat_mul(q1, q2):
    """Multiply two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_inv(q):
    """Inverse of a unit wxyz quaternion."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


class IKPoseSolver:
    """IK via PyBullet — no external service needed, requires pybullet installed."""

    def __init__(
        self,
        env,
        *,
        joint_lower: list[float],
        joint_upper: list[float],
        joint_rest: list[float],
        debug: bool = False,
    ):
        import pybullet as p
        import pybullet_data
        self._p = p
        self._env = env
        self._lower = list(joint_lower)
        self._upper = list(joint_upper)
        self._rest  = list(joint_rest)
        self._ranges = [u - l for l, u in zip(self._lower, self._upper)]
        self._debug = debug
        # Each IKPoseSolver gets its own pybullet client (DIRECT mode).
        # CRITICAL: store the clientId and pass it to every p.* call —
        # otherwise pybullet defaults to the LAST connected client, which
        # contaminates IK across multiple RolloutEnv episodes.
        self._client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client_id)
        urdf_path = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
        self._robot = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=self._client_id)
        self._joint_ids = [
            j for j in range(p.getNumJoints(self._robot, physicsClientId=self._client_id))
            if p.getJointInfo(self._robot, j, physicsClientId=self._client_id)[2] != p.JOINT_FIXED
        ][:7]
        self._eef_id = p.getNumJoints(self._robot, physicsClientId=self._client_id) - 1
        # Frame offset: q_offset = q_pb^-1 * q_robo (computed lazily on first call)
        self._q_offset: np.ndarray | None = None
        self._q_offset_inv: np.ndarray | None = None

    def _compute_frame_offset(self, current_joints: np.ndarray) -> None:
        """Compute the constant rotation offset between pybullet EEF and robosuite EEF frames.

        pybullet's panda_grasptarget and robosuite's gripper0_right_eef share the same
        physical rotation but use different local axis conventions. The offset is constant
        for any joint configuration.
        """
        p = self._p
        cid = self._client_id
        for jid, q in zip(self._joint_ids, current_joints):
            p.resetJointState(self._robot, jid, q, physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)
        ls = p.getLinkState(self._robot, self._eef_id, computeForwardKinematics=True, physicsClientId=cid)
        q_pb_xyzw = np.array(ls[1])
        q_pb = np.array([q_pb_xyzw[3], q_pb_xyzw[0], q_pb_xyzw[1], q_pb_xyzw[2]])

        # Robosuite EEF orientation (wxyz)
        _, q_robo = self._env.eef_pose_base()

        # Offset: q_robo = q_pb * q_offset  →  q_offset = q_pb^-1 * q_robo
        self._q_offset = _quat_mul(_quat_inv(q_pb), q_robo)
        self._q_offset_inv = _quat_inv(self._q_offset)

    def solve(self, target_pos: np.ndarray, target_quat: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        p = self._p
        cid = self._client_id
        for jid, q in zip(self._joint_ids, current_joints):
            p.resetJointState(self._robot, jid, q, physicsClientId=cid)

        if self._q_offset is None:
            self._compute_frame_offset(current_joints)

        q_pb_target = _quat_mul(target_quat, self._q_offset_inv)
        quat_xyzw = [q_pb_target[1], q_pb_target[2], q_pb_target[3], q_pb_target[0]]
        joints = p.calculateInverseKinematics(
            self._robot, self._eef_id, target_pos.tolist(), quat_xyzw,
            lowerLimits=self._lower, upperLimits=self._upper,
            jointRanges=self._ranges, restPoses=self._rest,
            maxNumIterations=200, residualThreshold=1e-6,
            physicsClientId=cid,
        )
        joints_out = np.array(joints[:7], dtype=np.float64)

        if self._debug:
            for jid, q in zip(self._joint_ids, joints_out):
                p.resetJointState(self._robot, jid, q, physicsClientId=cid)
            ls = p.getLinkState(self._robot, self._eef_id, computeForwardKinematics=True, physicsClientId=cid)
            q_pb_fk_xyzw = np.array(ls[1])
            q_pb_fk = np.array([q_pb_fk_xyzw[3], q_pb_fk_xyzw[0], q_pb_fk_xyzw[1], q_pb_fk_xyzw[2]])
            q_robo_pred = _quat_mul(q_pb_fk, self._q_offset)
            print(f"  [ik]  joints        : {joints_out.round(3)}")
            print(f"  [ik]  q_pb_target   : {q_pb_target.round(3)}")
            print(f"  [ik]  q_pb FK after : {q_pb_fk.round(3)}")
            print(f"  [ik]  q_robo target : {target_quat.round(3)}")
            print(f"  [ik]  q_robo pred   : {q_robo_pred.round(3)}")

        return joints_out
