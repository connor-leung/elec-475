"""Run sketch_lab.ipynb headless with some CFG settings changed, and save it with its outputs.

The overrides are written into the Settings cell itself (not passed through the environment), so
the saved notebook shows exactly which settings produced its outputs.

    python tools/run_notebook.py OUT.ipynb KEY=VALUE ...
    python tools/run_notebook.py executed/base_run.ipynb
    python tools/run_notebook.py executed/aug.ipynb "AUG='strong'" "RUN_TAG='aug'"
"""
import json
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

LAB = Path(__file__).resolve().parent.parent
out_path = Path(sys.argv[1])
overrides = [a.split('=', 1) for a in sys.argv[2:]]

nb = nbformat.read(LAB / 'sketch_lab.ipynb', as_version=4)
cfg_cell = next(c for c in nb.cells if c.cell_type == 'code' and c.source.startswith('CFG = dict('))
for key, value in overrides:
    pattern = rf'^(\s*){key}=[^,\n]*,'
    if not re.search(pattern, cfg_cell.source, flags=re.M):
        sys.exit(f'{key} is not a setting in the CFG cell')
    cfg_cell.source = re.sub(pattern, lambda m: f'{m.group(1)}{key}={value},', cfg_cell.source,
                             count=1, flags=re.M)

out_path.parent.mkdir(parents=True, exist_ok=True)
t0 = time.time()
client = NotebookClient(nb, timeout=None, kernel_name='python3',
                        resources={'metadata': {'path': str(LAB)}})
failed = None
try:
    client.execute()
except CellExecutionError as e:
    failed = e
finally:
    nbformat.write(nb, out_path)
print(f'saved {out_path} after {(time.time() - t0) / 60:.1f} min')
if failed:
    print('FAILED:', str(failed)[-3000:])
    sys.exit(1)
