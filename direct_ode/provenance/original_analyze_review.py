"""Post-review re-scoring of immutable saved outputs; no fitting or simulation."""
from pathlib import Path
import hashlib,json,shutil,math
import pandas as pd
import numpy as np

B=Path(__file__).resolve().parent;R=B.parent;V=R/'manuscript_revision_v2';D=B/'data'
for folder in ['data','figures']:
    shutil.copytree(V/folder,B/folder,dirs_exist_ok=True)
for name in ['build_documents.py','make_figures.py','render_with_word.ps1','check_render.py','captions.json']:
    if not (B/name).exists():(B/name).write_text((V/name).read_text(encoding='utf-8').replace('revision_v2','revision_v3'),encoding='utf-8')
review=Path(r'C:\Users\osmar\.codex\attachments\18b1bcd3-8f92-4a32-b827-dc5a3a6dae14\Pasted text.txt')
shutil.copyfile(review,B/'references/AI_presubmission_review.txt')
sources={}
def read(name,**kw):
    path=R/name;sources[name]=hashlib.sha256(path.read_bytes()).hexdigest();return pd.read_csv(path,**kw)
C=['San Diego','Los Angeles','San Francisco']
W=read('phase18b_projections_v2/projections_weekly.csv',parse_dates=['origin','week_start'])
q=W[W.scenario.isin(['P0','P1','P2','P3','P4'])].copy()
idx=['county','model','delay','origin','h','week_start']
pred=q.pivot(index=idx,columns='scenario',values='pred')
obs=q[q.scenario=='P0'].set_index(idx)[['obs','naive_pred']]
p=pred.join(obs).reset_index()
assert p[['P0','P1','P2','P3','P4']].notna().all().all()
p['lo']=p[['P0','P1','P2','P3','P4']].min(axis=1);p['hi']=p[['P0','P1','P2','P3','P4']].max(axis=1)
p['lead_days']=(p.week_start-p.origin).dt.days
p=p[p.lead_days>0].copy();assert ((p.lead_days-1)%7==0).all()
p['future_week']=(p.lead_days-1)//7+1
p['covered']=(p.obs>=p.lo)&(p.obs<=p.hi)
p['p0_abs_error']=(p.P0-p.obs).abs();p['naive_abs_error']=(p.naive_pred-p.obs).abs()
p.to_csv(D/'future_week_targets.csv',index=False)
rows=[];byweek=[]
for (c,m,de),z in p.groupby(['county','model','delay']):
    for window,lo,hi in [('future1_4',1,4),('future5_7',5,7),('future1_7',1,7)]:
        s=z[z.future_week.between(lo,hi)];v=s[s.obs.notna()]
        rows.append(dict(county=c,model=m,delay=int(de),window=window,n=int(len(v)),covered=int(v.covered.sum()),coverage=float(v.covered.mean()),missing=int(s.obs.isna().sum()),missing_forecasts=43*(hi-lo+1)-len(s),n_origins=int(s.origin.nunique()),p0_mae=float(v.p0_abs_error.mean()),naive_mae=float(v.naive_abs_error.mean())))
    for h,s in z[z.future_week<=7].groupby('future_week'):
        v=s[s.obs.notna()];byweek.append(dict(county=c,model=m,delay=int(de),future_week=int(h),n=len(v),covered=int(v.covered.sum()),coverage=float(v.covered.mean())))
F=pd.DataFrame(rows);F.to_csv(D/'future_week_summary.csv',index=False);pd.DataFrame(byweek).to_csv(D/'future_week_by_lead.csv',index=False)
# Calendar alignment is verified directly, not assumed from reporting index h.
# Missing known admission weeks can shift the old h grid by an additional week.
common=[];unmatched=[]
for (c,origin),z in p[p.future_week<=7].groupby(['county','origin']):
    sets=[set(s.week_start) for _,s in z.groupby(['model','delay'])];intersection=set.intersection(*sets)
    common.append(z[z.week_start.isin(intersection)])
    if any(s!=sets[0] for s in sets):unmatched.append(dict(county=c,origin=str(origin.date()),shared_weeks=len(intersection),calendar_sets={str(key):[str(v.date()) for v in s.week_start] for key,s in z.groupby(['model','delay'])}))
matched=pd.concat(common,ignore_index=True);matched.to_csv(D/'future_week_matched_targets.csv',index=False)
mr=[]
for (c,m,de),z in matched.groupby(['county','model','delay']):
    for window,lo,hi in [('future1_4',1,4),('future5_7',5,7),('future1_7',1,7)]:
        s=z[z.future_week.between(lo,hi)];v=s[s.obs.notna()]
        mr.append(dict(county=c,model=m,delay=int(de),window=window,n=len(v),covered=int(v.covered.sum()),coverage=float(v.covered.mean())))
pd.DataFrame(mr).to_csv(D/'future_week_matched_summary.csv',index=False)
(D/'future_week_calendar_exceptions.json').write_text(json.dumps(unmatched,indent=2),encoding='utf-8')

S=read('phase17b_layer3b/serology_check.csv',parse_dates=['date'])
S=S[S.used_for=='validation'].copy()
bands=[('Original asymmetric',-.10,.20),('Symmetric 0.05',-.05,.05),('Symmetric 0.10',-.10,.10),('Symmetric 0.20',-.20,.20)]
thresholds=[.5,2/3,.8];serorows=[];errors=[]
for c in C:
    for period in ['All dates','2020-2021','2022']:
        s=S[S.county==c]
        if period=='2020-2021':s=s[s.date.dt.year<=2021]
        elif period=='2022':s=s[s.date.dt.year==2022]
        for model,key in [('L3','L3_phase17_ever_compartments'),('L3B','L3B_ever_compartments')]:
            err=s[key]-s.sero
            errors.append(dict(county=c,period=period,model=model,n=len(s),mean_abs_diff=float(err.abs().mean()),median_abs_diff=float(err.abs().median()),mean_signed_diff=float(err.mean())))
            for band,lo,hi in bands:
                within=int(((err>=lo-1e-12)&(err<=hi+1e-12)).sum());frac=within/len(s)
                for threshold in thresholds:
                    serorows.append(dict(county=c,period=period,model=model,band=band,lo=lo,hi=hi,required_fraction=threshold,n=len(s),within=within,fraction=frac,met=frac>=threshold-1e-12))
T=pd.DataFrame(serorows);T.to_csv(D/'serology_threshold_sensitivity.csv',index=False);pd.DataFrame(errors).to_csv(D/'serology_residual_summary.csv',index=False)
for c in C:
    r=T[(T.county==c)&(T.period=='All dates')&(T.model=='L3B')&(T.band=='Original asymmetric')&(T.required_fraction==2/3)].iloc[0]
    assert int(r.within)=={'San Diego':19,'Los Angeles':20,'San Francisco':17}[c]

A=read('phase18_scenarios/reconstruction.csv')
A=A[(A.model=='L3B')&(A.delay==1)]
rr=[]
for _,r in A.iterrows():
    for tol in [.1,.2,.3]:
        for days in [7,14,21]:
            total=abs(float(r.ratio)-1)<=tol+1e-12;peak=abs(float(r.peak_shift_days))<=days
            rr.append(dict(county=r.county,period=r.period,total_tolerance=tol,peak_tolerance_days=days,total_ratio=float(r.ratio),peak_shift_days=float(r.peak_shift_days),total_met=total,peak_met=peak,original_serology_met=bool(r.R2),joint_met=bool(total and peak and r.R2)))
pd.DataFrame(rr).to_csv(D/'reconstruction_threshold_sensitivity.csv',index=False)

# q changes across fits and stitched weeks, though no separate severity process is modelled.
P=read('phase17b_layer3b/params_states.csv',parse_dates=['origin'])
P=P[(P.model=='L3B')&(P.delay==1)].copy();P['first_week_date']=P.origin-pd.Timedelta(days=56)
P[['county','origin','first_week_date','ihr']].to_csv(D/'admissions_q_history.csv',index=False)
qr=[]
for c in C:
    for period,start,end in [('behaviour','2020-10-03','2021-04-02'),('vaccination','2020-12-12','2021-12-03')]:
        z=P[(P.county==c)&P.first_week_date.between(start,end)]
        qr.append(dict(county=c,period=period,n=int(len(z)),q_min=float(z.ihr.min()),q_median=float(z.ihr.median()),q_max=float(z.ihr.max())))
pd.DataFrame(qr).to_csv(D/'admissions_q_summary.csv',index=False)
sigma=1/5.1;delta=1/7
t=np.linspace(0,80,801);k=sigma*delta/(sigma-delta)*(np.exp(-delta*t)-np.exp(-sigma*t))
pd.DataFrame({'days_since_exposure':t,'kernel_density':k}).to_csv(D/'admissions_kernel.csv',index=False)
assert abs(float(np.trapezoid(k,t))-1)<1e-4
kernel=dict(sigma=sigma,delta=delta,mean_days=1/sigma+1/delta,sd_days=math.sqrt(1/sigma**2+1/delta**2),interpretation='Model-implied continuous-time delay from entry into E to I outflow, with fixed transition rates; not an independently estimated clinical delay.')

out=dict(date='2026-10-17',plan_sha256=hashlib.sha256((B/'V3_ANALYSIS_PLAN.txt').read_bytes()).hexdigest(),review_sha256=hashlib.sha256(review.read_bytes()).hexdigest(),method='Post hoc descriptive re-scoring of saved paths and decision thresholds; no model fitting or simulation.',kernel=kernel,calendar_exceptions=unmatched,source_sha256=sources,future_primary=F[(F.model=='L3B')&(F.delay==1)].to_dict('records'),serology_primary=T[(T.period=='All dates')&(T.model=='L3B')&(T.required_fraction==2/3)].to_dict('records'),q_summaries=qr)
(D/'v3_analysis_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('WHOLLY FUTURE PRIMARY\n'+F[(F.model=='L3B')&(F.delay==1)].to_string(index=False))
print('\nSEROLOGY 2/3 RULE\n'+T[(T.period=='All dates')&(T.model=='L3B')&(T.required_fraction==2/3)][['county','band','within','n','met']].to_string(index=False))
print('\nMATCHED RESIDUALS\n'+pd.DataFrame(errors)[pd.DataFrame(errors).period=='All dates'].to_string(index=False))
print('\nQ RANGE\n'+pd.DataFrame(qr).to_string(index=False))
