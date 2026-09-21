from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import numpy as np


@dataclass
class CameraConfig:
    name: str = "frontview"
    width: int = 512
    height: int = 512


@dataclass
class RandomizationConfig:
    """Per-trial domain randomization options (applies to all tasks)."""
    # Object placement
    x_range: list[float] = field(default_factory=lambda: [-0.18, 0.18])
    y_range: list[float] = field(default_factory=lambda: [-0.12, 0.12])
    rotation: bool = True          # randomize tabletop object z-rotation on each reset
    # Visual
    colors: bool = False           # randomize non-robot geom colors
    lighting: bool = False         # randomize light position and intensity
    seed: int | None = None        # fixed seed for reproducibility (None = vary each trial)


@dataclass
class SimConfig:
    task: str = "can_in_bin"
    motion_speed: float = 1.0   # 1.0 = full speed; 0.5 = half speed (more waypoints)
    camera: CameraConfig = field(default_factory=CameraConfig)
    camera_views: list[str] = field(default_factory=list)
    """If non-empty, one camera is randomly selected from this list at the start of each trial."""
    visualize: bool = False
    use_mjviewer: bool = False
    """Use robosuite's mjviewer (interactive free camera) instead of the OpenCV viewer. Only for    
    interactive debugging — mjviewer contends with the offscreen renderer during long
    blocking calls."""
    object_aliases: dict[str, str] = field(default_factory=dict)
    """Maps prompt names → simulator body names, e.g. {"red cube": "cubeA"}."""
    perception: Literal["auto", "gt", "sam3"] = "auto"
    """Perception backend: 'gt' = ground-truth sim state, 'sam3' = SAM3 service, 'auto' = SAM3 if reachable else gt."""
    grasp: Literal["pca", "graspgen"] = "pca"
    """Grasp backend: 'pca' = PCA-based top-down grasp (no external server), 'graspgen' = GraspGen 6-DOF service."""
    randomize: RandomizationConfig = field(default_factory=RandomizationConfig)


@dataclass
class RobotConfig:
    ik_backend: str = "pybullet"
    sam3_url: str | None = "http://127.0.0.1:8114"
    sam3_debug: bool = False
    """Save SAM3 segmentation overlays to {output_dir}/debug_sam3/ for inspection."""
    bbox_debug: bool = False
    """Save segmented pointcloud and AABB as .ply to {output_dir}/debug_bbox/ for inspection."""
    motion_debug: bool = False
    """Print IK target/FK and joint-controller tracking residuals each move."""
    graspgen_url: str | None = "tcp://127.0.0.1:5556"
    graspgen_config: str | None = None
    """Path to GraspGen gripper config YAML (e.g. graspgen_franka_panda.yml). Required when grasp='graspgen'."""
    graspgen_venv: str | None = None
    """Path to GraspGen .venv directory. Defaults to third_party/GraspGen/.venv."""
    graspgen_debug: bool = False
    """Save all GraspGen grasp candidates + selected grasp + point cloud as .ply to {output_dir}/debug_graspgen/."""
    graspgen_horizontal: bool = False
    """Favour GraspGen grasps whose closing axis is horizontal (both fingers at the same z).
    Good for small knobs/handles where fingers should wrap left/right rather than top/bottom."""
    graspgen_vertical: bool = False
    """Favour GraspGen grasps whose approach is top-down (good for small objects on a table)."""
    tcp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    grasp_stall_delta: float = 2e-4   # finger joint delta below which grasp stalled
    grasp_steps_per_iter: int = 10    # sim steps per gripper increment
    grasp_max_iters: int = 20
    # Arm joint limits (rad). Defaults are the Franka Panda spec — override per
    # robot when adding new arms. `joint_rest` is the null-space bias for IK.
    joint_lower: list[float] = field(default_factory=lambda: [
        -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    joint_upper: list[float] = field(default_factory=lambda: [
         2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    joint_rest: list[float] = field(default_factory=lambda: [
         0.0,    -0.785,   0.0,    -2.356,   0.0,     1.571,   0.785 ])
    # Home joint configuration (rad). Defaults are Franka Panda init_qpos.
    home_joints: list[float] = field(default_factory=lambda: [
         0.0, -np.pi/36,   0.0, -np.pi/2 - np.pi/3,   0.0, np.pi - 0.49, np.pi/4 ])


@dataclass
class LLMConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.7
    max_tokens: int = 2048
    model: str = "openai/gpt-5.4"
    api_key_env: str = "OPENROUTER_API_KEY"


@dataclass
class CollectionConfig:
    record_video: bool = False
    video_fps: int = 20
    model_frame: Literal["table"] = "table"
    sim: SimConfig = field(default_factory=SimConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    task_name: str = "Place the can in the bin."
    turn_budget: int = 20
    max_retries: int = 2
    stop_on_reward: bool = True
    """If True, end the episode as soon as reward >= 1.0 and give the VLM one
    closing turn. Set False to require the VLM to terminate itself — the episode
    then runs until it emits a text-only turn or the turn budget is exhausted."""
    total_trials: int = 100
    num_workers: int = 1
    output_dir: str = "data/collect"
