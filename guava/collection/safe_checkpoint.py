"""Read only the small known type set in GUAVA's generated NumPy checkpoints."""
import pickle
import gzip
import hashlib
import io
from pathlib import Path
import re
import numpy as np
from guava.collection.checkpoint import Checkpoint


class CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('guava.collection.checkpoint', 'Checkpoint'): Checkpoint,
            ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype,
            ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy._core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy.core.numeric', '_frombuffer'): np.core.numeric._frombuffer,
            ('numpy._core.numeric', '_frombuffer'): np.core.numeric._frombuffer,
            ('numpy.core.multiarray', 'scalar'): np.core.multiarray.scalar,
            ('numpy._core.multiarray', 'scalar'): np.core.multiarray.scalar,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported checkpoint type: {module}.{name}')
        return allowed[module, name]


def load_checkpoint(path):
    with open(path, 'rb') as stream:
        compressed = stream.read(2) == b'\x1f\x8b'
        stream.seek(0)
        if compressed:
            with gzip.GzipFile(fileobj=stream, mode='rb') as decoded:
                ck = CheckpointUnpickler(decoded).load()
        else:
            ck = CheckpointUnpickler(stream).load()
    if not isinstance(ck, Checkpoint):
        raise ValueError('Not a GUAVA checkpoint')
    snapshot = getattr(ck, 'snapshot', None)
    if snapshot and 'model_sha256' in snapshot:
        model_hash = snapshot['model_sha256']
        if not re.fullmatch(r'[0-9a-f]{64}', model_hash):
            raise ValueError('Invalid checkpoint model hash')
        asset = Path(path).parent / 'models' / f'{model_hash}.pkl.gz'
        blob = gzip.decompress(asset.read_bytes())
        if hashlib.sha256(blob).hexdigest() != model_hash:
            raise ValueError('Checkpoint model asset hash mismatch')
        model = CheckpointUnpickler(io.BytesIO(blob)).load()
        if not isinstance(model, dict) or set(model) not in ({'xml', 'model_arrays'}, {'xml', 'model_binary'}):
            raise ValueError('Invalid checkpoint model asset')
        snapshot.update(model)
    for name in ('qpos', 'qvel', 'ctrl', 'current_joints', 'eef_pos'):
        a = getattr(ck, name)
        if not isinstance(a, np.ndarray) or a.dtype.kind not in 'fi' or not np.isfinite(a).all():
            raise ValueError(f'Invalid checkpoint field {name}')
    return ck
