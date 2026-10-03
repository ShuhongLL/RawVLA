#!/usr/bin/env python3
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--output',required=True); p.add_argument('--run-tag',required=True); p.add_argument('--tasks',required=True)
p.add_argument('--episodes',type=int,required=True); p.add_argument('--start-episode',type=int,required=True)
p.add_argument('--eval-start-seed',default='')
p.add_argument('--eval-seeds',default=''); p.add_argument('--match-dataset-seeds',default='0')
p.add_argument('--save-every',type=int,required=True); p.add_argument('--original-image-key',required=True)
p.add_argument('--policy-image-key',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--result-dir',required=True)
a=p.parse_args()
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
payload=vars(a)|{'mode':'easy','task_config':'demo_clean_rgb_eval','created_utc':datetime.now(timezone.utc).isoformat(),
 'camera_views':['head_camera','left_camera','right_camera'],'action_source':'StarVLA model prediction via websocket; training expert joint paths are not loaded',
 'psnr_semantics':'same-index policy rollout render versus clean expert dataset RGB; trajectory divergence affects this diagnostic'}
out.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
