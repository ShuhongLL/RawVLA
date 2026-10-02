#!/usr/bin/env python3
"""Create reproducible RoboTwin2 easy policy job plans; never submits jobs."""
import argparse,json,re
from datetime import datetime,timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument('--output-dir',required=True); p.add_argument('--batch-size',type=int,default=20)
p.add_argument('--episodes',type=int,default=50); p.add_argument('--code-root',default=str(REPO_ROOT))
a=p.parse_args(); out=Path(a.output_dir)
if out.exists(): raise SystemExit(f'Refusing to overwrite {out}')
out.mkdir(parents=True)
s=(Path(__file__).parent/'run_robotwin2_starvla_rgb_eval.sh').read_text()
tasks=[x.strip() for x in re.search(r'TASKS=\(\n(.*?)\n\)',s,re.S).group(1).splitlines() if x.strip()]
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); rows=[]
for i,task in enumerate(tasks):
    batch=i//a.batch_size+1; tag=f'easy50_batch{batch:02d}_{task}_{stamp}'
    result=str(Path(a.code_root) / 'results' / 'robotwin2-policy-closed-loop' / tag)
    cmd=(f'cd {a.code_root} && TASK={task} EPISODES={a.episodes} SAVE_EVERY=5 '
         f'ORIGINAL_IMAGE_KEY=rgb POLICY_IMAGE_KEY=default_isp RUN_TAG={tag} '
         'bash scripts/run_robotwin2_policy_easy.sh')
    item={'batch':batch,'task':task,'run_tag':tag,'result_dir':result,'result_exists':Path(result).exists(),
          'resources':{'gpu_type':'NVIDIA-L4','gpu_count':1,'priority':9},'job_command':cmd}
    rows.append(item); (out/f'{task}.job.json').write_text(json.dumps(item,indent=2)+'\n')
(out/'plan.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'submitted':False,'jobs':rows},indent=2)+'\n')
with (out/'commands.sh').open('w') as f:
    f.write('# Prepared commands only. Review credentials and run a 1x1 queue smoke before submission.\n')
    for r in rows: f.write(f"# batch {r['batch']}\n{r['job_command']}\n")
print(json.dumps({'tasks':len(tasks),'batches':max(r['batch'] for r in rows),'sizes':[sum(r['batch']==b for r in rows) for b in range(1,max(r['batch'] for r in rows)+1)],'output':str(out)}))
