"""Continue the already queued preparation with training and verification."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

out=Path(__file__).resolve().parent
os.environ['PYTHONPATH']='scripts;src'
while True:
    status=json.loads((out/'state.json').read_text(encoding='utf8'))['status']
    if status=='prepared':break
    if status not in ('waiting_gpu','encoding'):raise RuntimeError('Preparation did not complete: '+status)
    time.sleep(10)
for script,log in [('train_replay.py','train.log'),('report.py','report.log')]:
    with (out/log).open('w',encoding='utf8') as stream:
        subprocess.run([sys.executable,'-u',str(out/script)],stdout=stream,stderr=subprocess.STDOUT,check=True)
print('Training, replay, report and archive all complete.',flush=True)
