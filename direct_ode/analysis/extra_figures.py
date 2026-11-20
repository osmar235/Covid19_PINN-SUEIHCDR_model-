"""New review-driven figures and replacement of the scenario-evaluation figure."""
from pathlib import Path
import json,math
import numpy as np
import pandas as pd

# Reuse the same drawing functions and regenerate the inherited figure series.
import make_figures as G
from reportlab.graphics.shapes import Drawing,Rect
B=Path(__file__).resolve().parent;D=B/'data';C=G.C

# Main 5: preserve the old grid and show the newly scored future grid separately.
old=pd.read_csv(D/'envelope_by_h.csv');new=pd.read_csv(D/'future_week_by_lead.csv')
d=Drawing(500,534);G.legend(d,515,[(c,z,'line') for c,z in zip(C,[G.BLUE,G.ORANGE,G.GREEN])],x=82)
for y,title,xmax,xlabel,data,key in [(336,'A  Original reporting week evaluation',8,'Original reporting index h',old,'h'),(135,'B  Wholly future calendar week evaluation',7,'Future week from first Sunday after issue',new,'future_week')]:
    ax=G.Ax(d,53,y,421,125,title,(1,xmax),(0,100),range(1,xmax+1),[0,25,50,75,100],xlabel=xlabel,ylabel='Observed targets bracketed (%)')
    ax.hline(50,G.GRAY,[4,2])
    for c,z in zip(C,[G.BLUE,G.ORANGE,G.GREEN]):
        q=data[data.county==c]
        if 'model' in q:q=q[(q.model=='L3B')&(q.delay==1)]
        ax.plot(q[key],100*q.coverage,z,points=True)
G.text(d,15,79,'C  Nominal primary grid at a Saturday issue date',9.6,True)
G.box(d,32,37,199,27,['Old h 1  |  Sunday -6 to Saturday 0'],fill='#FBEFE7',size=8.4)
G.box(d,240,37,225,27,['Old h 2  =  future week 1  |  days +1 to +7'],fill='#E9F2F8',size=8.2)
G.text(d,250,21,'Calendar dates determine eligibility, including exceptions after missing reports.',8.3,anchor='middle',color=G.GRAY)
G.text(d,250,5,'The dashed line retains the original 50% reference; the new analysis is post hoc.',8.3,anchor='middle',color=G.GRAY)
G.save(d,'fig5_origin_scenarios')

# S3: report decision-rule dependence without selecting a replacement threshold.
s=pd.read_csv(D/'serology_threshold_sensitivity.csv');r=pd.read_csv(D/'reconstruction_threshold_sensitivity.csv')
bands=['Symmetric 0.05','Symmetric 0.10','Original asymmetric','Symmetric 0.20']
d=Drawing(500,451);G.legend(d,434,[(c,z,'line') for c,z in zip(C,[G.BLUE,G.ORANGE,G.GREEN])],x=82)
ax=G.Ax(d,53,267,421,115,'A  L3B donor dates within each tolerance band',(0.7,4.3),(0,105),[1,2,3,4],[0,25,50,75,100],ylabel='Donor dates within band (%)',xformat=lambda x:['+/-0.05','+/-0.10','-0.10 to +0.20','+/-0.20'][int(x)-1])
for t in [50,100*2/3,80]:ax.hline(t,G.GRAY,[2,2])
for c,z in zip(C,[G.BLUE,G.ORANGE,G.GREEN]):
    v=s[(s.county==c)&(s.period=='All dates')&(s.model=='L3B')&np.isclose(s.required_fraction,2/3)].set_index('band')
    ax.plot([1,2,3,4],[100*v.loc[b,'fraction'] for b in bands],z,points=True)
G.text(d,250,233,'Reference fractions: 50%, two thirds and 80%. Each county has 21 donor dates.',8.3,anchor='middle',color=G.GRAY)
G.text(d,20,204,'B  Joint reconstruction decisions under alternative rules',9.7,True)
labels=['San Diego  vaccination','Los Angeles  vaccination','San Francisco  vaccination','San Diego  behaviour','Los Angeles  behaviour','San Francisco  behaviour']
combos=[(t,days) for t in [.1,.2,.3] for days in [7,14,21]]
x0=176;cw=33;top=152;rh=20
for j,t in enumerate([10,20,30]):G.text(d,x0+(j*3+1.5)*cw,183,f'Total +/-{t}%',8.7,True,anchor='middle')
for j,(t,days) in enumerate(combos):G.text(d,x0+(j+.5)*cw,168,str(days)+' d',8.4,anchor='middle')
for i,label in enumerate(labels):
    c=C[i%3];period='vaccination' if i<3 else 'behaviour';y=top-(i+1)*rh
    G.text(d,20,y+6,label,8.5)
    for j,(t,days) in enumerate(combos):
        q=r[(r.county==c)&(r.period==period)&np.isclose(r.total_tolerance,t)&(r.peak_tolerance_days==days)].iloc[0]
        good=bool(q.joint_met)
        d.add(Rect(x0+j*cw,y,cw,rh,fillColor=G.col('#E7F2EF' if good else '#F7EAE5'),strokeColor=G.col('#FFFFFF'),strokeWidth=1))
        G.text(d,x0+(j+.5)*cw,y+6,'P' if good else 'F',8.6,True,anchor='middle')
d.add(Rect(x0+4*cw,top-6*rh,cw,6*rh,fillColor=None,strokeColor=G.col(G.BLACK),strokeWidth=1.1))
G.text(d,250,12,'P = joint rule met; F = not met. The outlined column is the original +/-20%, 14 day rule.',8.2,anchor='middle',color=G.GRAY)
G.save(d,'figS3_thresholds')

# S4: clarify fixed delay structure and already time-varying fitted q.
k=pd.read_csv(D/'admissions_kernel.csv');p=pd.read_csv(D/'admissions_q_history.csv',parse_dates=['first_week_date'])
d=Drawing(500,546)
ax=G.Ax(d,54,410,420,88,'A  Model implied delay from exposure to I outflow',(0,60),(0,.07),[0,15,30,45,60],[0,.02,.04,.06],ylabel='Delay density per day',xlabel='Days since entry into E')
z=k[k.days_since_exposure<=60];ax.plot(z.days_since_exposure,z.kernel_density,G.BLUE);ax.vline(12.1,G.ORANGE,[3,2])
G.text(d,267,477,'Mean 12.1 days; fixed across variants and windows',8.5,color=G.ORANGE)
aa,ticks,fmt=G.dateaxis('2020-10-03',['2020-10-03','2021-04-01','2021-10-01','2021-12-03'])
for i,c in enumerate(C):
    q=p[(p.county==c)&p.first_week_date.between('2020-10-03','2021-12-03')].sort_values('first_week_date')
    ax=G.Ax(d,54,279-i*111,420,70,f'{chr(66+i)}  {c}  |  Fitted admissions multiplier',(0,426),(0,6),ticks,[0,2,4,6],ylabel='q (%)',xformat=fmt)
    xx=[];yy=[]
    for j,row in enumerate(q.itertuples()):
        x=(row.first_week_date-aa).days;xx.extend([x,min(x+7,426)]);yy.extend([100*row.ihr,100*row.ihr])
    ax.plot(xx,yy,[G.BLUE,G.ORANGE,G.GREEN][i])
G.text(d,250,12,'q is constant within a window, varies between fits, and is stitched by week in reconstruction.',8.3,anchor='middle',color=G.GRAY)
G.save(d,'figS4_admission_mapping')
print('Added future-week evaluation, threshold sensitivity and admission-mapping figures.')
