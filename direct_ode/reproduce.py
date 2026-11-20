"""Replay archived results into a NEW directory; never overwrite frozen results.

Run from the release root with Python 3.14 and requirements-model.txt installed:
    python -B reproduce.py --out replay/clean_check
"""
from pathlib import Path
import argparse,datetime as dt,hashlib,json,subprocess,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compare(expected,actual):
    rows=[]
    for p in sorted(expected.iterdir()):
        if not p.is_file() or p.suffix not in ['.csv','.json'] or p.name=='run_manifest.json':continue
        q=actual/p.name
        if not q.exists():raise AssertionError('Missing replay output '+str(q))
        exact=sha(p)==sha(q)
        if p.suffix=='.csv':
            a=pd.read_csv(p);b=pd.read_csv(q)
            pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)
        else:
            a=json.loads(p.read_text());b=json.loads(q.read_text())
            def check(x,y):
                if isinstance(x,dict):
                    assert x.keys()==y.keys()
                    for k in x:check(x[k],y[k])
                elif isinstance(x,list):
                    assert len(x)==len(y)
                    for i,j in zip(x,y):check(i,j)
                elif isinstance(x,(int,float)) and not isinstance(x,bool):
                    assert np.isclose(x,y,rtol=1e-10,atol=1e-10,equal_nan=True),(x,y)
                else:assert x==y,(x,y)
            check(a,b)
        rows.append({'file':p.name,'bytes_identical':exact,'numerically_equivalent':True,
                     'expected_sha256':sha(p),'replay_sha256':sha(q)})
    return rows
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='replay/clean_check')
    ap.add_argument('--verify-existing',action='store_true',help='Recheck completed replay outputs and regenerate arithmetic tables, without repeating simulations')
    args=ap.parse_args()
    out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT):raise SystemExit('Output must remain inside release root')
    if out.exists() and any(out.iterdir()) and not args.verify_existing:raise SystemExit('Use a fresh empty replay directory')
    if args.verify_existing and not out.is_dir():raise SystemExit('The replay directory must exist')
    out.mkdir(parents=True,exist_ok=True)
    # Validate every raw/derived input included in the release inventory.
    for item in json.loads((ROOT/'provenance/input_inventory.json').read_text()):
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    commands=[
      ('hhs_admissions_v2.py',['--out',str(out/'phase15_admissions')],'phase15_admissions'),
      ('projections_phase18b.py',['--out',str(out/'phase18b_projections_v2')],'phase18b_projections_v2'),
      ('scenarios_phase18.py',['--out',str(out/'phase18_scenarios')],'phase18_scenarios'),
      ('scenarios_phase18.py',['--fits','phase19_layer3t','--name','L3T','--out',str(out/'phase19_scenarios')],'phase19_scenarios'),
      ('test_phase18b.py',['--out',str(out/'phase18b_tests')],None)
    ]
    results=[]
    for script,argv,folder in commands:
        command=[sys.executable,'-B',script]+argv
        print('RUN',script,' '.join(argv),flush=True)
        label=folder or 'phase18b_tests'
        if args.verify_existing:
            log=(out/(label+'_console.txt')).read_text(encoding='utf-8')
            if folder is None:assert 'Summary: 18 PASS, 0 FAIL' in log
            print('VERIFY existing completed',label,flush=True)
        else:
            run=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            (out/(label+'_console.txt')).write_text(run.stdout,encoding='utf-8')
            if run.returncode:
                print(run.stdout);raise SystemExit(run.returncode)
            print(run.stdout[-850:],flush=True)
        outputs=compare(ROOT/folder,out/folder) if folder else []
        manifest=out/label/'run_manifest.json'
        environment={}
        if manifest.exists():
            data=json.loads(manifest.read_text())
            environment={k:data.get(k) for k in ['python','numpy','pandas','scipy','statsmodels','platform'] if k in data}
        results.append({'script':script,'arguments':argv,'returncode':0,'outputs':outputs,'simulation_environment':environment})
    # These scripts use only saved outputs and arithmetic. They create analysis/data.
    for script in ['analysis/build_evidence.py','analysis/rescore_saved_results.py']:
        run=subprocess.run([sys.executable,'-B',script],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (out/(Path(script).stem+'_console.txt')).write_text(run.stdout,encoding='utf-8')
        if run.returncode:print(run.stdout);raise SystemExit(run.returncode)
    expected=ROOT/'expected_paper_tables';actual=ROOT/'analysis/data'
    table_checks=[]
    for p in sorted(expected.glob('*.csv')):
        q=actual/p.name
        if not q.exists():raise AssertionError('Missing derived table '+p.name)
        a=pd.read_csv(p);b=pd.read_csv(q)
        pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)
        table_checks.append({'file':p.name,'numerically_equivalent':True,'bytes_identical':sha(p)==sha(q)})
    report={'utc':dt.datetime.now(dt.timezone.utc).isoformat(),'verification_python':sys.version,
            'rtol':1e-10,'atol':1e-10,'model_refits_performed':False,
            'scope':'Fresh admissions processing, all saved-fit projections and reconstructions; paper tables. Estimation fits reused, not repeated.',
            'runs':results,'paper_table_checks':table_checks,'passed':True}
    (out/'replay_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'replayed_result_files':sum(len(x['outputs']) for x in results),'paper_tables':len(table_checks),'report':str(out/'replay_verification.json')},indent=2))
if __name__=='__main__':main()
