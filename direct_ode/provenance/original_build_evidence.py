"""Assemble manuscript evidence from existing outputs; no fitting or simulation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA = BASE / 'data'
DATA.mkdir(exist_ok=True)
COUNTIES = ['San Diego', 'Los Angeles', 'San Francisco']
SOURCES = {}

def track(path):
    p = ROOT / path
    SOURCES[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def read(path, **kwargs):
    return pd.read_csv(track(path), **kwargs)

def js(path):
    return json.loads(track(path).read_text(encoding='utf-8'))

def primary(d, name='L3B'):
    return d[(d['model'] == name) & (d['delay'] == 1)].copy()

crit = {n:js(f'{folder}/criteria.json') for n,folder in [('L2','phase16_layer2'),('L3','phase17_layer3'),('L3B','phase17b_layer3b'),('L3T','phase19_layer3t'),('P18','phase18_scenarios'),('P18B','phase18b_projections_v2')]}
sero = read('phase17b_layer3b/serology_check.csv')
sero = sero[sero['used_for'] == 'validation'].copy()
sero.to_csv(DATA/'serology_validation.csv',index=False)
recon = primary(read('phase18_scenarios/reconstruction.csv'))
recon.to_csv(DATA/'reconstruction.csv',index=False)
recon19 = primary(read('phase19_scenarios/reconstruction.csv'),'L3T')
recon19.to_csv(DATA/'reconstruction_testing.csv',index=False)
scenario_all = read('phase18_scenarios/scenario_summary.csv')
scenario = primary(scenario_all)
scenario.to_csv(DATA/'scenarios.csv',index=False)
rows=[]
for c in COUNTIES:
    for s in ['V0','V1','V2','B0','B1']:
        q=scenario_all[(scenario_all.county==c)&(scenario_all.scenario==s)]
        b=q[(q.model=='L3B')&(q.delay==1)]
        if len(b): rows.append(dict(county=c,scenario=s,primary=float(b.iloc[0].pct_admissions),minimum=float(q.pct_admissions.min()),maximum=float(q.pct_admissions.max())))
pd.DataFrame(rows).to_csv(DATA/'scenario_sensitivity.csv',index=False)
forecast=[]
for model,folder in [('L2','phase16_layer2'),('L3','phase17_layer3'),('L3B','phase17b_layer3b'),('L3T','phase19_layer3t')]:
    q=primary(read(f'{folder}/summary.csv'),model)
    for _,r in q.iterrows(): forecast.append(dict(county=r['county'],model=model,n_origins=int(r.n_origins),mean_error=float(r.mean_err)))
q=read('phase17b_layer3b/summary.csv')
for _,r in q[(q.delay==1)&q.model.isin(['ARIMA_ADM','CASELAG','NAIVE_ADM'])].iterrows(): forecast.append(dict(county=r['county'],model=r.model,n_origins=int(r.n_origins),mean_error=float(r.mean_err)))
pd.DataFrame(forecast).to_csv(DATA/'forecast_errors.csv',index=False)
params=primary(read('phase17b_layer3b/params_states.csv'))
params['first_week_month']=(pd.to_datetime(params.origin)-pd.Timedelta(days=56)).dt.strftime('%Y-%m')
dec=params[params.first_week_month=='2020-12'].groupby('county')[['mult_E','mult_I']].agg(lambda s:float(np.exp(np.log(s).mean())))
dec.to_csv(DATA/'december_state_corrections.csv')
envelope=read('phase18b_projections_v2/envelope_e1.csv')
envelope.to_csv(DATA/'envelope_all_chains.csv',index=False)
w=primary(read('phase18b_projections_v2/projections_weekly.csv'))
w=w[w.scenario.isin(['P0','P1','P2','P3','P4'])]
p=w.pivot_table(index=['county','origin','h','obs'],columns='scenario',values='pred').reset_index()
p['lo']=p[['P0','P1','P2','P3','P4']].min(axis=1)
p['hi']=p[['P0','P1','P2','P3','P4']].max(axis=1)
p['covered']=(p.obs>=p.lo)&(p.obs<=p.hi)
p.groupby(['county','h']).agg(n=('covered','size'),covered=('covered','sum'),coverage=('covered','mean')).reset_index().to_csv(DATA/'envelope_by_h.csv',index=False)
coverage=[]
for c in COUNTIES:
    z=p[p.county==c]
    for win,sel in [('h1_4',z.h<=4),('h5_8',z.h>=5)]:
        x=z[sel]; coverage.append(dict(county=c,window=win,n=len(x),covered=int(x.covered.sum()),coverage=float(x.covered.mean())))
pd.DataFrame(coverage).to_csv(DATA/'envelope_counts.csv',index=False)
drift=primary(read('phase18b_projections_v2/drift_e2.csv'))
drift.to_csv(DATA/'drift.csv',index=False)
hindsight=primary(read('phase18b_projections_v2/hindsight_vax_e3.csv'))
hindsight.to_csv(DATA/'hindsight.csv',index=False)
daily=primary(read('phase18_scenarios/scenario_daily.csv'))
daily[daily.scenario=='S0'].to_csv(DATA/'reconstruction_daily.csv',index=False)
for c in COUNTIES:
    f='phase15_admissions/weekly_'+c.replace(' ','')+'.csv'
    read(f).to_csv(DATA/('observed_'+c.replace(' ','')+'.csv'),index=False)
for f in ['d3_lags_peaks.csv','d4_adm_per_case.csv','d5_penalty.csv']:
    read('phase18_diagnosis/'+f).to_csv(DATA/f,index=False)
for f in ['PHASE16_PROTOCOL.md','PHASE17_PROTOCOL.md','PHASE17B_PROTOCOL.md','PHASE18_PROTOCOL.md','PHASE19_PROTOCOL.md','PHASE18B_PROTOCOL.md','layer2_admissions.py','layer3b_sero.py','layer3t_tests.py','scenarios_phase18.py','projections_phase18b.py','literature/literature_sources_2026-10-14.json']:
    track(f)
audits=[]
for folder in ['phase17b_layer3b','phase18_scenarios','phase19_layer3t','phase19_scenarios','phase18b_projections_v2']:
    m=js(folder+'/run_manifest.json'); bad=[]
    outputs=m.get('outputs_sha256',m.get('output_sha256',{}))
    for f,h in outputs.items():
        path=ROOT/folder/f
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=h: bad.append(f)
    if bad: raise ValueError((folder,bad))
    audits.append(dict(folder=folder,checked_outputs=len(outputs),mismatches=bad))
E=dict(criteria=crit,december_corrections=dec.reset_index().to_dict('records'),coverage_counts=coverage,manifest_audits=audits,source_sha256=SOURCES,method='Read-only extraction and arithmetic summaries of existing saved outputs. No model code imported, no fitting, simulation, bootstrap or significance tests rerun.')
(DATA/'evidence.json').write_text(json.dumps(E,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(dict(december_corrections=E['december_corrections'],coverage_counts=coverage,manifest_audits=audits,observed_columns=list(pd.read_csv(DATA/'observed_SanDiego.csv').columns)),indent=2))
