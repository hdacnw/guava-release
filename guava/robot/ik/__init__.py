from guava.config import RobotConfig


def get_ik_solver(cfg: RobotConfig, env, ik_mode: str = "pybullet"):
    from guava.robot.ik.ik_pose import IKPoseSolver
    return IKPoseSolver(
        env,
        joint_lower=cfg.joint_lower,
        joint_upper=cfg.joint_upper,
        joint_rest=cfg.joint_rest,
        debug=cfg.motion_debug,
    )
