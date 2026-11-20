"""Publication figures drawn from saved outputs, with editable SVG masters."""
from pathlib import Path
import math,json
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing,String,Line,Rect,PolyLine,Polygon,Circle,Group
from reportlab.graphics import renderSVG,renderPDF
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium

import os
def font_path(var,candidates):
    path=os.environ.get(var)
    if path and Path(path).is_file():return path
    for p in candidates:
        if Path(p).is_file():return p
    raise FileNotFoundError('Set '+var+' to an installed TrueType font path')
pdfmetrics.registerFont(TTFont('FigureArial',font_path('FIGURE_FONT_REGULAR',[r'C:\Windows\Fonts\arial.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'])))
pdfmetrics.registerFont(TTFont('FigureArialBold',font_path('FIGURE_FONT_BOLD',[r'C:\Windows\Fonts\arialbd.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'])))

B=Path(__file__).resolve().parent; D=B/'data'; O=B/'figures'
O.mkdir(exist_ok=True)
C=['San Diego','Los Angeles','San Francisco']
BLUE='#0072B2'; ORANGE='#D55E00'; GREEN='#009E73'; PURPLE='#8657A3'; BLACK='#252525'; GRAY='#707070'; LIGHT='#E9EEF2'
def col(x):return HexColor(x)
def text(d,x,y,s,size=9,bold=False,color=BLACK,anchor='start'):
    d.add(String(x,y,str(s),fontName='FigureArialBold' if bold else 'FigureArial',fontSize=size,fillColor=col(color),textAnchor=anchor))
def line(d,x1,y1,x2,y2,color=BLACK,width=.9,dash=None):
    d.add(Line(x1,y1,x2,y2,strokeColor=col(color),strokeWidth=width,strokeDashArray=dash or []))
def dot(d,x,y,color,r=2.2):d.add(Circle(x,y,r,fillColor=col(color),strokeColor=None))
def box(d,x,y,w,h,lines,fill='#F2F5F7',stroke='#A8B4BD',size=9):
    d.add(Rect(x,y,w,h,rx=5,ry=5,fillColor=col(fill),strokeColor=col(stroke),strokeWidth=.7))
    for i,s in enumerate(lines):text(d,x+w/2,y+h/2+(len(lines)-1)*6-12*i-3,s,size,bold=i==0,anchor='middle')
def arrow(d,x1,y1,x2,y2,label='',color=GRAY,dash=None):
    line(d,x1,y1,x2,y2,color,1,dash)
    a=math.atan2(y2-y1,x2-x1);sz=4.5
    pts=[x2,y2,x2-sz*math.cos(a-.5),y2-sz*math.sin(a-.5),x2-sz*math.cos(a+.5),y2-sz*math.sin(a+.5)]
    d.add(Polygon(pts,fillColor=col(color),strokeColor=None))
    if label:text(d,(x1+x2)/2,(y1+y2)/2+5,label,8,anchor='middle',color=color)
def legend(d,y,items,x=47):
    for label,color,style in items:
        if style=='dot':dot(d,x+7,y,color,2)
        else:line(d,x,y,x+15,y,color,1.6,[4,2] if style=='dash' else None)
        text(d,x+20,y-3,label,8.6);x+=20+len(label)*4.25+15
class Ax:
    def __init__(self,d,x,y,w,h,title,xlim,ylim,xticks,yticks,xlabel='',ylabel='',xformat=None,yformat=None):
        self.d=d;self.x=x;self.y=y;self.w=w;self.h=h;self.xlim=xlim;self.ylim=ylim
        text(d,x,y+h+13,title,9.4,True)
        for v in yticks:
            z=self.py(v);line(d,x,z,x+w,z,'#DFE4E8',.45)
            text(d,x-6,z-2.8,yformat(v) if yformat else f'{v:g}',8.1,anchor='end',color=GRAY)
        for v in xticks:
            z=self.px(v);line(d,z,y,z,y-3,GRAY,.5)
            text(d,z,y-14,xformat(v) if xformat else f'{v:g}',8.1,anchor='middle',color=GRAY)
        line(d,x,y,x+w,y,GRAY,.6);line(d,x,y,x,y+h,GRAY,.6)
        if xlabel:text(d,x+w/2,y-29,xlabel,8.5,anchor='middle')
        if ylabel:
            g=Group();g.translate(x-34,y+h/2);g.rotate(90)
            g.add(String(0,0,ylabel,fontName='FigureArial',fontSize=8.5,textAnchor='middle',fillColor=col(BLACK)));d.add(g)
    def px(self,v):return self.x+(v-self.xlim[0])/(self.xlim[1]-self.xlim[0])*self.w
    def py(self,v):return self.y+(v-self.ylim[0])/(self.ylim[1]-self.ylim[0])*self.h
    def plot(self,xx,yy,color=BLUE,width=1.3,dash=None,points=False,only=False):
        pairs=[(float(x),float(y)) for x,y in zip(xx,yy) if np.isfinite(x) and np.isfinite(y)]
        coords=[v for x,y in pairs for v in (self.px(x),self.py(y))]
        if len(coords)>3 and not only:self.d.add(PolyLine(coords,strokeColor=col(color),strokeWidth=width,strokeDashArray=dash or [],fillColor=None))
        if points or only:
            for x,y in pairs:dot(self.d,self.px(x),self.py(y),color,1.9)
    def hline(self,y,color=GRAY,dash=None):line(self.d,self.x,self.py(y),self.x+self.w,self.py(y),color,.8,dash)
    def vline(self,x,color=GRAY,dash=None):line(self.d,self.px(x),self.y,self.px(x),self.y+self.h,color,.8,dash)
    def band(self,lo,hi,fill='#EDF5F0'):
        # Transparent-looking flat shading; redraw reference lines above it.
        self.d.add(Rect(self.x,self.py(lo),self.w,self.py(hi)-self.py(lo),fillColor=col(fill),strokeColor=None))
    def whisker(self,x,lo,hi,color=BLUE):
        xx=self.px(x);y1=self.py(lo);y2=self.py(hi)
        line(self.d,xx,y1,xx,y2,color,1.3);line(self.d,xx-3,y1,xx+3,y1,color,1.3);line(self.d,xx-3,y2,xx+3,y2,color,1.3)
def dateaxis(start,ticks):
    a=pd.Timestamp(start);return a,[(pd.Timestamp(t)-a).days for t in ticks],lambda v:(a+pd.Timedelta(days=v)).strftime('%b %y')
def save(d,name):
    svg=O/(name+'.svg');pdf=O/(name+'.pdf')
    renderSVG.drawToFile(d,str(svg))
    s=svg.read_text(encoding='utf-8').replace('font-family: FigureArialBold;','font-family: Arial; font-weight: bold;').replace('font-family: FigureArial;','font-family: Arial;')
    svg.write_text(s,encoding='utf-8')
    renderPDF.drawToFile(d,str(pdf))
    doc=pdfium.PdfDocument(str(pdf));doc[0].render(scale=300/72).to_pil().save(O/(name+'.png'));doc.close()

# 1. Mechanistic structure and distinct questions asked of it.
d=Drawing(500,485)
text(d,15,467,'A  Transmission and observation model',11,True)
box(d,25,315,76,43,['Susceptible','S'])
box(d,159,315,65,43,['Exposed','E'])
box(d,283,315,65,43,['Infectious','I'])
box(d,403,306,82,61,['Recovered','Rpre  R1  R2'])
box(d,18,399,90,39,['Vaccine protected','Uv'],fill='#E8F2F8')
box(d,18,228,90,39,['Behaviour protected','Ub'],fill='#E8F4EF')
arrow(d,101,337,159,337,'infection')
arrow(d,224,337,283,337,'progression')
arrow(d,348,337,403,337,'outflow')
arrow(d,45,358,45,399);text(d,13,377,'doses',7.7)
arrow(d,83,399,83,358);text(d,91,377,'waning',7.7)
arrow(d,45,315,45,267);text(d,5,290,'risk signal',7.7)
arrow(d,83,267,83,315);text(d,91,290,'relaxation',7.7)
arrow(d,108,414,180,358,'escape',BLUE)
line(d,444,367,444,414,BLUE,.9);line(d,444,414,204,414,BLUE,.9)
arrow(d,204,414,204,358,'',BLUE);text(d,318,420,'variant dependent reinfection',8.3,color=BLUE,anchor='middle')
box(d,159,231,90,37,['Reported cases','ascertainment'],fill='#FBF4E9')
box(d,294,231,100,37,['Admissions','infection severity'],fill='#FBF4E9')
arrow(d,253,333,211,268,'',ORANGE,[3,2]);arrow(d,375,333,345,268,'',ORANGE,[3,2])
text(d,250,209,'Solid arrows show state flows; dashed arrows show observations.',8.4,anchor='middle',color=GRAY)
text(d,15,179,'B  Evaluation addresses distinct uses of the fitted model',11,True)
for x,title,subt in [(15,'Fit and infection scale',['Cases and admissions','Separate donor survey']),
                     (180,'Continuous reconstruction',['One initial state','No later E/I corrections']),
                     (345,'Origin specific scenarios',['43 issue dates per county','Five conditional paths'])]:
    box(d,x,91,140,66,[title]+subt,size=8.8)
arrow(d,155,124,180,124);arrow(d,320,124,345,124)
text(d,250,66,'114 weekly fits per county  |  56 day estimation windows  |  Three counties',8.8,anchor='middle')
text(d,250,43,'Passing one assessment does not imply success at the next.',9.4,True,anchor='middle',color=BLUE)
text(d,250,22,'Study design schematic; arrows between assessments are not a validation theorem.',8.2,anchor='middle',color=GRAY)
save(d,'fig1_model_evaluation')

# 2. Survey observations and residuals make the agreement band visible.
sero=pd.read_csv(D/'serology_validation.csv',parse_dates=['date'])
a,ticks,fmt=dateaxis('2020-07-01',['2020-07-01','2021-07-01','2022-07-01','2022-12-01']);end=(pd.Timestamp('2022-12-01')-a).days
d=Drawing(500,530)
legend(d,513,[('Donor survey',BLACK,'dot'),('L3 before calibration',ORANGE,'dash'),('L3B calibrated',BLUE,'line')],x=40)
for i,c in enumerate(C):
    q=sero[sero.county==c].sort_values('date');xx=(q.date-a).dt.days;y=363-i*155
    ax=Ax(d,43,y,191,108,f'{chr(65+2*i)}  {c}',(0,end),(0,1),ticks,[0,.25,.5,.75,1],ylabel='Ever infected share',xformat=fmt)
    ax.plot(xx,q.L3_phase17_ever_compartments,ORANGE,dash=[4,2]);ax.plot(xx,q.L3B_ever_compartments,BLUE);ax.plot(xx,q.sero,BLACK,only=True)
    ax=Ax(d,294,y,186,108,f'{chr(66+2*i)}  Model minus donor survey',(0,end),(-.45,.65),ticks,[-.4,-.2,0,.2,.4,.6],ylabel='Difference in share',xformat=fmt)
    ax.band(-.1,.2);ax.hline(-.1,GREEN,[3,2]);ax.hline(.2,GREEN,[3,2]);ax.hline(0,GRAY,[2,2])
    ax.plot(xx,q.L3_phase17_ever_compartments-q.sero,ORANGE,dash=[4,2]);ax.plot(xx,q.L3B_ever_compartments-q.sero,BLUE,points=True)
text(d,250,16,'Shading marks the declared tolerance of -0.10 to +0.20; it is not an uncertainty interval.',8.4,anchor='middle')
save(d,'fig2_serology')

# 3. Both complete reconstruction periods, using the same scale within a county.
daily=pd.read_csv(D/'reconstruction_daily.csv',parse_dates=['date'])
rec=pd.read_csv(D/'reconstruction.csv')
d=Drawing(500,542);legend(d,525,[('Observed',BLACK,'dot'),('October start',ORANGE,'line'),('December start',BLUE,'line')],x=65)
for i,c in enumerate(C):
    obs=pd.read_csv(D/('observed_'+c.replace(' ','')+'.csv'),parse_dates=['week_start'])
    vals=[]
    for period in ['behaviour','vaccination']:
        q=daily[(daily.county==c)&(daily.period==period)]
        vals+=list(q.adm)
    ymax=math.ceil(1.18*max(vals)/(10**math.floor(math.log10(max(vals)))))*(10**math.floor(math.log10(max(vals))))
    for j,(period,start,end,tt) in enumerate([('behaviour','2020-10-03','2021-04-02',['2020-10-03','2021-01-01','2021-04-01']),('vaccination','2020-12-12','2021-12-03',['2020-12-12','2021-06-01','2021-12-01'])]):
        aa,ticks,fmt=dateaxis(start,tt);q=daily[(daily.county==c)&(daily.period==period)];rr=rec[(rec.county==c)&(rec.period==period)].iloc[0]
        oo=obs[(obs.week_start>=start)&((obs.week_start+pd.Timedelta(days=6))<=end)&obs.adm_per_day.notna()]
        y=366-i*156;x=44+j*250
        ax=Ax(d,x,y,188,107,f'{chr(65+2*i+j)}  {c}  |  {"October start" if j==0 else "December start"}',(0,(pd.Timestamp(end)-aa).days),(0,ymax),ticks,np.linspace(0,ymax,5),ylabel='Admissions per day',xformat=fmt)
        ax.plot((q.date-aa).dt.days,q.adm,ORANGE if j==0 else BLUE)
        ax.plot((oo.week_start-aa).dt.days+3,oo.adm_per_day,BLACK,only=True)
        text(d,x+5,y+92,f'Total ratio {rr.ratio:.3f}; peak {rr.peak_shift_days:+.0f} days',8.1,color=BLACK)
text(d,250,15,'County scales are shared across periods. The October and December runs use different states and inputs.',8.2,anchor='middle')
save(d,'fig3_reconstruction')

# 4. Boundary corrections displayed against the dates of the corrected first weeks.
p=pd.read_csv(D/'state_corrections.csv',parse_dates=['first_week_date'])
aa,ticks,fmt=dateaxis('2020-10-01',['2020-10-01','2020-12-01','2021-02-01','2021-04-01'])
p=p[(p.first_week_date>=aa)&(p.first_week_date<='2021-04-02')]
d=Drawing(500,515)
text(d,250,496,'Corrections at successive overlapping window boundaries',10,True,anchor='middle')
for i,c in enumerate(C):
    q=p[p.county==c].sort_values('first_week_date')
    for j,(key,label,color) in enumerate([('mult_E','Exposed',BLUE),('mult_I','Infectious',ORANGE)]):
        y=354-i*151;x=44+j*250
        ax=Ax(d,x,y,188,103,f'{chr(65+2*i+j)}  {c}  |  {label}',(0,183),(.6,1.25),ticks,[.6,.8,1,1.2],ylabel='State multiplier',xformat=fmt)
        x0=ax.px((pd.Timestamp('2020-12-01')-aa).days);x1=ax.px((pd.Timestamp('2021-01-01')-aa).days)
        d.add(Rect(x0,y,x1-x0,103,fillColor=col('#EAF0F5'),strokeColor=None))
        ax.hline(1,GRAY,[3,2]);ax.plot((q.first_week_date-aa).dt.days,q[key],color,points=True)
text(d,250,17,'Shading marks December 2020. A multiplier below one reduces the carried E or I state.',8.5,anchor='middle')
save(d,'fig4_state_corrections')

# 5. Coverage, drift and the partly elapsed first target.
env=pd.read_csv(D/'envelope_by_h.csv');drift=pd.read_csv(D/'drift.csv')
d=Drawing(500,534);legend(d,515,[(c,z,'line') for c,z in zip(C,[BLUE,ORANGE,GREEN])],x=82)
ax=Ax(d,53,336,421,130,'A  Bracketing by the five prescribed scenarios',(1,8),(0,100),range(1,9),[0,25,50,75,100],xlabel='Reporting week index h',ylabel='Observed targets covered (%)')
ax.hline(50,GRAY,[4,2])
for c,z in zip(C,[BLUE,ORANGE,GREEN]):
    q=env[env.county==c];ax.plot(q.h,100*q.coverage,z,points=True)
text(d,468,ax.py(50)+5,'Declared threshold',8.3,anchor='end',color=GRAY)
ax=Ax(d,53,135,421,125,'B  Status quo ratios among realized growth targets',(1,8),(0,4),range(1,9),[0,1,2,3,4],xlabel='Reporting week index h',ylabel='Median projected / observed')
ax.hline(1,GRAY,[4,2])
for c,z in zip(C,[BLUE,ORANGE,GREEN]):
    q=drift[(drift.county==c)&(drift.phase_definition=='realized_phase')&(drift.phase=='growth')].sort_values('h');ax.plot(q.h,q.median_ratio_p0_over_obs,z,points=True)
text(d,15,80,'C  Primary reporting grid at an example Saturday origin',9.6,True)
box(d,45,35,178,28,['h 1  |  Sunday -6 to Saturday 0'],fill='#FBEFE7',size=8.6)
box(d,231,35,185,28,['h 2  |  Sunday +1 to Saturday +7'],fill='#E9F2F8',size=8.6)
line(d,217,31,217,69,BLACK,1.1);text(d,217,20,'Origin',8.6,anchor='middle')
text(d,132,20,'Six fitted days + one projected day',8.1,anchor='middle',color=GRAY)
text(d,250,3,'Scenario range is deterministic; it has no specified probability level.',8.2,anchor='middle',color=GRAY)
save(d,'fig5_origin_scenarios')

# 6. Retrospective vaccination timing and conditional effect sizes.
v=pd.read_csv(D/'vaccination_daily.csv',parse_dates=['date']);sc=pd.read_csv(D/'scenario_sensitivity.csv')
d=Drawing(500,535)
legend(d,518,[('Baseline',BLUE,'line'),('Doses 28 days earlier',GREEN,'line'),('San Francisco pace',PURPLE,'dash')],x=36)
aa,ticks,fmt=dateaxis('2020-12-12',['2020-12-12','2021-06-01','2021-12-01'])
for i,c in enumerate(C):
    q=v[v.county==c];mx=q.adm.max();unit=10**math.floor(math.log10(mx));ym=math.ceil(mx/unit)*unit
    ax=Ax(d,48,380-i*111,426,71,f'{chr(65+i)}  {c}',(0,356),(0,ym),ticks,[0,ym/2,ym],ylabel='Admissions/day',xformat=fmt)
    for s,z,dash in [('S0',BLUE,None),('V2',GREEN,None),('V1',PURPLE,[4,2])]:
        if c=='San Francisco' and s=='V1':continue
        qq=q[q.scenario==s];ax.plot((qq.date-aa).dt.days,qq.adm,z,dash=dash)
ax=Ax(d,48,35,188,79,'D  No subsequent doses',(0,4),(0,700),[1,2,3],[0,200,400,600],ylabel='Admissions change (%)',xformat=lambda x:['SD','LA','SF'][int(x)-1])
for i,c in enumerate(C):
    r=sc[(sc.county==c)&(sc.scenario=='V0')].iloc[0];ax.whisker(i+1,r.minimum,r.maximum,ORANGE);dot(d,ax.px(i+1),ax.py(r.primary),ORANGE,2.7)
    text(d,ax.px(i+1)+7,ax.py(r.primary)-3,f'+{r.primary:.0f}%',8,color=ORANGE)
ax=Ax(d,296,35,178,79,'E  Earlier or faster vaccination',(0,4),(-45,0),[1,2,3],[-40,-20,0],ylabel='Admissions change (%)',xformat=lambda x:['SD','LA','SF'][int(x)-1])
for i,c in enumerate(C):
    for s,z,shift in [('V2',GREEN,-.11),('V1',PURPLE,.11)]:
        r=sc[(sc.county==c)&(sc.scenario==s)]
        if r.empty:continue
        r=r.iloc[0];ax.whisker(i+1+shift,r.minimum,r.maximum,z);dot(d,ax.px(i+1+shift),ax.py(r.primary),z,2.6)
text(d,250,4,'Whiskers span selected model chains; they are sensitivity ranges, not confidence intervals.',8.2,anchor='middle',color=GRAY)
save(d,'fig6_vaccination')

# S1. Saved forecast comparisons, kept distinct from reconstruction and intervention claims.
f=pd.read_csv(D/'forecast_errors.csv');e=json.loads((D/'evidence.json').read_text())
d=Drawing(500,448)
models=['NAIVE_ADM','ARIMA_ADM','CASELAG','L2','L3','L3B','L3T']
labels=['Persistence','ARIMA','Case lag','L2','L3','L3B','L3T']
for i,c in enumerate(C):
    q=f[f.county==c].set_index('model');mx=max(q.mean_error)*1.12
    x=56+i*153
    ax=Ax(d,x,232,123,152,f'{chr(65+i)}  {c}',(0,mx),(-.6,6.6),[0,mx/2,mx],[0,1,2,3,4,5,6],xlabel='MAE admissions/day',yformat=lambda z:labels[6-int(z)],xformat=lambda z:f'{z:.1f}')
    for k,m in enumerate(models):
        yy=6-k;zz=BLUE if m=='L3B' else GREEN if m=='L3T' else GRAY
        dot(d,ax.px(q.loc[m,'mean_error']),ax.py(yy),zz,2.5)
text(d,40,176,'D  Paired L3B minus ARIMA mean error with saved 95% bootstrap interval',9.5,True)
for i,c in enumerate(C):
    v=e['criteria']['L3B']['D1'][c];y=135-i*32
    text(d,43,y-3,c,8.6);lo,hi=v['ci95'];mean=v['mean_diff'];x0=190;scale=1.32
    line(d,x0+lo*scale,y,x0+hi*scale,y,BLUE,1.4);dot(d,x0+mean*scale,y,BLUE,2.8)
    line(d,x0,y-8,x0,y+8,GRAY,.7,[2,2]);text(d,485,y+8,f'{mean:.2f} [{lo:.2f}, {hi:.2f}]',8.4,anchor='end')
line(d,177,43,454,43,GRAY,.6)
for tick in [0,50,100,150,200]:
    xx=190+tick*1.32;line(d,xx,43,xx,40,GRAY,.6);text(d,xx,29,tick,8,anchor='middle',color=GRAY)
text(d,250,17,'Intervals are from the saved four origin block bootstrap; crossing zero does not establish equivalence.',8.1,anchor='middle')
save(d,'figS1_forecast')

# S2. A targeted sensitivity can improve forecast errors without repairing reconstruction.
r=pd.read_csv(D/'reconstruction_testing.csv');r0=pd.read_csv(D/'reconstruction.csv')
allenv=pd.read_csv(D/'envelope_all_chains.csv')
d=Drawing(500,340);legend(d,321,[('Primary L3B',BLUE,'dot'),('Testing linked L3T',GREEN,'dot')],x=100)
ax=Ax(d,51,185,423,88,'A  October start reconstruction after the testing sensitivity',(0,4),(.8,2.2),[1,2,3],[.8,1,1.2,1.6,2,2.2],ylabel='Model / observed total',xformat=lambda x:C[int(x)-1])
ax.band(.8,1.2);ax.hline(1.2,GRAY,[3,2]);ax.hline(1,GRAY,[2,2])
for i,c in enumerate(C):
    for q,z,sh in [(r0,BLUE,-.09),(r,GREEN,.09)]:
        rr=q[(q.county==c)&(q.period=='behaviour')].iloc[0];dot(d,ax.px(i+1+sh),ax.py(rr.ratio),z,3)
ax=Ax(d,51,40,423,88,'B  Early bracketing in all three saved projection chains',(0,4),(0,60),[1,2,3],[0,20,40,50,60],ylabel='Targets covered (%)',xformat=lambda x:C[int(x)-1])
ax.hline(50,GRAY,[3,2])
for i,c in enumerate(C):
    for model,delay,z,sh in [('L3B',1,BLUE,-.14),('L3B',0,ORANGE,0),('L3B_NOBEHAV',1,PURPLE,.14)]:
        row=allenv[(allenv.county==c)&(allenv.model==model)&(allenv.delay==delay)].iloc[0]
        dot(d,ax.px(i+1+sh),ax.py(100*row.cov_w1_4),z,2.8)
legend(d,6,[('Delay one',BLUE,'dot'),('Delay zero',ORANGE,'dot'),('No behaviour pool',PURPLE,'dot')],x=60)
save(d,'figS2_sensitivity')

print('Saved six main figures and two supplementary figures as editable SVG.')
