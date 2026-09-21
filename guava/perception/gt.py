import numpy as np
import viser.transforms as vtf


def _norm(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def _body_xpos(env, sim_name: str) -> np.ndarray:
    """Read body world position directly from sim.data.xpos, bypassing obs cache.

    Tries sim_name, then sim_name + '_main' (robosuite convention), then
    falls back to the obs dict if no matching body is found.
    """
    sim = env.rob.sim
    for candidate in (sim_name, sim_name + "_main"):
        try:
            body_id = sim.model.body_name2id(candidate)
            return np.array(sim.data.xpos[body_id])
        except Exception:
            pass
    # Fallback: read from obs (may be stale at episode start)
    obs = env.obs()
    key = sim_name + "_pos"
    if key in obs:
        return np.asarray(obs[key])
    raise ValueError(
        f"Body '{sim_name}' not found in sim (tried '{sim_name}_main') "
        f"and '{key}' not in obs."
    )


class GTSolver:
    """Ground-truth 3-D positions read directly from simulator state.

    Args:
        env: RoboEnv instance.
        aliases: Optional mapping from prompt names to simulator obs names,
                 e.g. {"red cube": "cubeA", "green cube": "cubeB"}.
                 Lookup tries the alias map first, then falls back to fuzzy
                 matching against the raw obs keys.
    """

    def __init__(self, env, aliases: dict[str, str] | None = None):
        self.env = env
        # Normalize both sides of the alias map once at construction time
        self._aliases: dict[str, str] = {
            _norm(k): v for k, v in (aliases or {}).items()
        }

    def get_position(self, name: str, **_) -> list[float]:
        # Resolve via alias map first
        q = _norm(name)
        sim_name = self._aliases.get(q)

        if sim_name is not None:
            pos_world = _body_xpos(self.env, sim_name)
        else:
            # Fuzzy match against MuJoCo body names
            sim = self.env.rob.sim
            all_bodies = [sim.model.body_id2name(i) for i in range(sim.model.nbody)]
            matches = [b for b in all_bodies if q in _norm(b)]
            # Prefer exact base-name match (e.g. 'cubeA' in 'cubeA_main')
            if not matches:
                raise ValueError(
                    f"No body matching '{name}'. "
                    f"Available bodies: {sorted(all_bodies)}. "
                    f"Tip: add an entry to sim.object_aliases in your config."
                )
            if len(matches) > 1:
                # Prefer the shortest match (most specific base body)
                matches = sorted(matches, key=len)
            body_id = sim.model.body_name2id(matches[0])
            pos_world = np.array(sim.data.xpos[body_id])

        base = vtf.SE3(wxyz_xyz=self.env.base_wxyz_xyz())
        obj  = vtf.SE3(wxyz_xyz=np.concatenate([[1, 0, 0, 0], pos_world]))
        return (base.inverse() @ obj).translation().tolist()
