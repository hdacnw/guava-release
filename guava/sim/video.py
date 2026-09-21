"""Optional, streaming episode video. Recording never advances simulation."""
from contextlib import contextmanager
from pathlib import Path
import json
import math


def add_video_arguments(parser):
    parser.add_argument('--record-video', action='store_true',
                        help='Save an MP4 of each episode from the active observation camera.')
    parser.add_argument('--video-fps', type=int, default=20,
                        help='Simulation-time playback FPS (1–60; default 20).')


def validate_video_fps(fps):
    if not isinstance(fps, int) or isinstance(fps, bool) or not 1 <= fps <= 60:
        raise ValueError('Video FPS must be an integer between 1 and 60')


class EpisodeVideo:
    def __init__(self, env, path, fps):
        import imageio.v2 as imageio
        validate_video_fps(fps)
        self.env, self.path, self.fps = env, Path(path), fps
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.path.with_suffix('.video.json').exists():
            raise FileExistsError(f'Refusing to overwrite video: {self.path}')
        self.start_time = float(env.rob.sim.data.time)
        self.frames = 0
        self.writer = imageio.get_writer(str(self.path), format='FFMPEG', fps=fps,
                                        codec='libx264', pixelformat='yuv420p',
                                        macro_block_size=2)

    def capture(self):
        self.writer.append_data(self.env.capture())
        self.frames += 1

    def on_step(self):
        # Control steps can be slower than video FPS. Repeat the current frame
        # for missed sample slots; never add physics steps to manufacture frames.
        elapsed = max(0., float(self.env.rob.sim.data.time) - self.start_time)
        target = math.floor(elapsed * self.fps + 1e-6) + 1
        if self.frames < target:
            frame = self.env.capture()
            while self.frames < target:
                self.writer.append_data(frame)
                self.frames += 1

    def close(self, completed):
        try:
            self.capture()  # Include the final state, even for a no-action episode.
        finally:
            self.writer.close()
        camera = self.env._cam_cfg or self.env.cfg.camera
        self.path.with_suffix('.video.json').write_text(json.dumps({
            'fps': self.fps, 'frames': self.frames, 'camera': camera.name,
            'simulation_start': self.start_time,
            'simulation_end': float(self.env.rob.sim.data.time),
            'completed_without_exception': completed,
            'timing': 'simulation time; model/API waiting is omitted',
        }, indent=2) + '\n')


@contextmanager
def record_episode(env, path, enabled=False, fps=20):
    if not enabled:
        yield
        return
    if getattr(env, '_video', None) is not None:
        raise RuntimeError('An episode recorder is already active')
    recorder = EpisodeVideo(env, path, fps)
    env._video = recorder
    completed = False
    try:
        recorder.capture()
        yield
        completed = True
    finally:
        env._video = None  # Never retain a writer in reused environments/checkpoints.
        recorder.close(completed)
