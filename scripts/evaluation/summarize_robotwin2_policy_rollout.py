#!/usr/bin/env python3
import argparse,csv,json,math
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('run_dir'); a=p.parse_args(); root=Path(a.run_dir)
results=[]
rp=root/'audit'/'episode_results.jsonl'
if rp.exists():
    results=[json.loads(x) for x in rp.read_text().splitlines() if x.strip()]
psnr={}; views={}
for f in (root/'audit').glob('*/episode_*/frame_psnr.csv'):
    for r in csv.DictReader(f.open()):
        v=float(r['psnr']); key=(r['task'],int(r['episode']))
        if math.isfinite(v): psnr.setdefault(key,[]).append(v); views.setdefault((key[0],key[1],r['view']),[]).append(v)
rows=[]
for r in results:
    key=(r['task'],int(r['episode'])); vals=psnr.get(key,[])
    row=dict(r); row['mean_psnr']=sum(vals)/len(vals) if vals else ''
    for view in ('head_camera','left_camera','right_camera'):
        vv=views.get((key[0],key[1],view),[]); row['mean_psnr_'+view]=sum(vv)/len(vv) if vv else ''
    rows.append(row)
fields=[
    'task','episode','eval_seed','success','num_steps','language','action_source',
    'policy_name','task_config','checkpoint','original_image_key','image_input_key',
    'result_dir','mean_psnr','mean_psnr_head_camera','mean_psnr_left_camera',
    'mean_psnr_right_camera',
]
with (root/'results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
with (root/'results.jsonl').open('w') as f:
    for r in rows: f.write(json.dumps(r)+'\n')
tasks=sorted({r['task'] for r in rows}); summary=[]
for task in tasks:
    rr=[r for r in rows if r['task']==task]; completed=len(rr); success=sum(bool(r['success']) for r in rr)
    def mean(field):
        x=[float(r[field]) for r in rr if r[field]!='']; return sum(x)/len(x) if x else ''
    summary.append({'task':task,'mode':'easy','episodes_attempted':completed,'episodes_completed':completed,'success_count':success,
      'success_rate':success/completed if completed else 0,'mean_psnr':mean('mean_psnr'),
      'mean_psnr_head_camera':mean('mean_psnr_head_camera'),'mean_psnr_left_camera':mean('mean_psnr_left_camera'),'mean_psnr_right_camera':mean('mean_psnr_right_camera'),'result_dir':str(root)})
sf=['task','mode','episodes_attempted','episodes_completed','success_count','success_rate','mean_psnr','mean_psnr_head_camera','mean_psnr_left_camera','mean_psnr_right_camera','result_dir']
with (root/'task_summary.csv').open('w',newline='') as f: w=csv.DictWriter(f,sf); w.writeheader(); w.writerows(summary)
total=len(rows); suc=sum(bool(r['success']) for r in rows); pv=[float(r['mean_psnr']) for r in rows if r['mean_psnr']!='']
overall={'total_tasks':len(tasks),'total_episodes':total,'total_success':suc,'overall_success_rate':suc/total if total else 0,'overall_mean_psnr':sum(pv)/len(pv) if pv else None}
(root/'overall_summary.json').write_text(json.dumps(overall,indent=2)+'\n')
print(json.dumps(overall,indent=2))
