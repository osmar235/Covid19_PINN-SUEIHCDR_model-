"""Compare the separate projection implementation on matching calendar targets.

The replication shares the original fitting/input modules and ODE integrator.
It checks projection implementation, not independent epidemiological validity.
Historical week-index differences are retained and reported, never imputed.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--replication',default='verification/replication_20261120')
    parser.add_argument('--out',default='verification/replication_comparison_20261120.json')
    args=parser.parse_args()
    rep=pd.read_csv(ROOT/args.replication/'projections_weekly.csv')
    rep=rep[rep['grid']=='p17b'].copy()
    rep['scenario']=rep['scenario'].replace({'H0':'P0_HINDSIGHT'})
    official=pd.read_csv(ROOT/'phase18b_projections_v2/projections_weekly.csv')
    official=official[(official['model']=='L3B') & (official['delay']==1)].copy()
    keys=['county','origin','scenario','week_start']
    assert not rep.duplicated(keys).any() and not official.duplicated(keys).any()
    joint=official.merge(rep,on=keys,how='outer',suffixes=('_official','_replication'),indicator=True)
    paired=joint[joint['_merge']=='both'].copy()
    paired['relative_difference']=(paired['pred_official']/paired['pred_replication']-1).abs()
    assert np.allclose(paired['pred_official'],paired['pred_replication'],rtol=1e-6,atol=1e-10)
    assert np.allclose(paired['obs_official'],paired['obs_replication'],equal_nan=True)
    unmatched=joint[joint['_merge']!='both'][keys+['_merge']].astype(str).to_dict('records')
    assert all(r['county']=='San Diego' and r['origin']=='2021-05-22' for r in unmatched)
    report={'passed':True,'primary_chain_only':True,'scenario_count':6,
            'paired_calendar_rows':len(paired),'max_relative_difference':float(paired['relative_difference'].max()),
            'max_relative_difference_by_scenario':paired.groupby('scenario')['relative_difference'].max().to_dict(),
            'unmatched_calendar_rows':unmatched,
            'explanation':'At San Diego origin 2021-05-22, the historical saved grid starts one week earlier. Six scenario rows on each boundary do not have matching calendar targets. They are excluded, not filled.',
            'scope':'Separate projection implementation using the same fitted states, input modules and ODE integrator. No refitting or external validation.'}
    out=ROOT/args.out;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
