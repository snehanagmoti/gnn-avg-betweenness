"""Run many (name, overrides) configs x seeds on 2 worker processes.
    python experiments/batch.py <step> '<json: {name: overrides}>' [seeds]"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
step, configs = sys.argv[1], json.loads(sys.argv[2])
seeds = sys.argv[3].split(",") if len(sys.argv) > 3 else ["0", "1", "2"]
jobs = [(n, o, s) for n, o in configs.items() for s in seeds]


def go(job):
    n, o, s = job
    r = subprocess.run([sys.executable, os.path.join(HERE, "lab.py"), n, json.dumps(o), step, s],
                       capture_output=True, text=True, cwd=os.path.dirname(HERE))
    print(r.stdout.strip() or r.stderr[-1500:], flush=True)


with ThreadPoolExecutor(int(os.environ.get("WORKERS", 2))) as ex:
    list(ex.map(go, jobs))
