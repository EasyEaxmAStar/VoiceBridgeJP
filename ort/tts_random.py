"""Utterance-scoped RNG restoration inside the exclusive TTS worker process."""
from contextlib import contextmanager
import os
import random
import threading
import numpy as np

_owner=threading.Lock()


@contextmanager
def isolated_utterance_rng(devices=None):
    import torch
    if devices is None:devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with _owner:
        python_state=random.getstate();numpy_state=np.random.get_state();hash_seed=os.environ.get('PYTHONHASHSEED')
        try:
            with torch.random.fork_rng(devices=devices):yield
        finally:
            random.setstate(python_state);np.random.set_state(numpy_state)
            if hash_seed is None:os.environ.pop('PYTHONHASHSEED',None)
            else:os.environ['PYTHONHASHSEED']=hash_seed
