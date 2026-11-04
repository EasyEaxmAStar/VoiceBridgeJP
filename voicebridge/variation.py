"""Per-preset semantic seed selection. No pitch, speed or sampling changes."""
from collections import Counter, deque
import json
from pathlib import Path
import secrets
import threading

LEVELS = ('STABLE',)


class SeedScheduler:
    def __init__(self, capacity=64):
        self.capacity = capacity
        self._lock = threading.Lock()
        self._history = {}

    def choose(self, preset, level, original, qualified=(), locked=None):
        if level not in LEVELS:
            raise ValueError('Unknown voice variation level')
        extra = list(qualified)
        pool = list(dict.fromkeys([*original, *(extra[:2] if level == LEVELS[1] else extra if level == LEVELS[2] else [])]))
        if not pool or any(type(s) is not int or not 0 <= s < 2**31 for s in pool):
            raise ValueError('Invalid qualified seed pool')
        with self._lock:
            history = self._history.setdefault(preset, deque(maxlen=self.capacity))
            if locked is not None:
                if locked not in pool:
                    raise ValueError('Locked seed is not qualified for this preset/variation level')
                selected = locked
            else:
                choices = [s for s in pool if not history or s != history[-1]] or pool
                if level != 'STABLE':
                    counts = Counter(history)
                    least = min(counts[s] for s in choices)
                    choices = [s for s in choices if counts[s] == least]
                    # Prefer the least recently used of equally used seeds.
                    recent = list(history)
                    ages = {s: (recent[::-1].index(s) if s in recent else self.capacity + 1) for s in choices}
                    choices = [s for s in choices if ages[s] == max(ages.values())]
                selected = secrets.choice(choices)
            history.append(selected)
            return selected


def qualified_seeds(preset):
    path = Path(__file__).resolve().parents[1] / 'config/voice_variation.qualified.json'
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding='utf8'))
    if data.get('schema') != 1:
        raise ValueError('Invalid seed qualification manifest')
    return data['presets'].get(preset, {}).get('seeds', [])
