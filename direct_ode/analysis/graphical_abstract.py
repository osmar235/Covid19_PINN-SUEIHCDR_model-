"""Separate figure files and a data-based graphical abstract using ReportLab."""
from pathlib import Path
import json,shutil
import pandas as pd
from reportlab.graphics.shapes import Drawing,String,Line,Rect
from reportlab.graphics import renderSVG,renderPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import pypdfium2 as pdfium

B=Path(__file__).resolve().parent;R=B.parent;P=B/'graphical_abstract';P.mkdir(exist_ok=True)
import os
def font_path(var,candidates):
 p=os.environ.get(var)
 if p and Path(p).is_file():return p
 return next(v for v in candidates if Path(v).is_file())
pdfmetrics.registerFont(TTFont('GARegular',font_path('FIGURE_FONT_REGULAR',[r'C:\Windows\Fonts\arial.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'])))
pdfmetrics.registerFont(TTFont('GABold',font_path('FIGURE_FONT_BOLD',[r'C:\Windows\Fonts\arialbd.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'])))
BLUE='#0072B2';ORANGE='#D55E00';GRAY='#777777';LIGHT='#DDDDDD';BLACK='#222222'
d=Drawing(800,320)
def text(x,y,t,size=12,color=BLACK,bold=False,anchor='start'):
 d.add(String(x,y,str(t),fontName='GABold' if bold else 'GARegular',fontSize=size,fillColor=HexColor(color),textAnchor=anchor))
def line(x,y,u,v,color=GRAY,dash=None):d.add(Line(x,y,u,v,strokeColor=HexColor(color),strokeWidth=.7,strokeDashArray=dash or []))
def rect(x,y,w,h,color):d.add(Rect(x,y,w,h,fillColor=HexColor(color),strokeColor=None))
text(400,299,'Serological agreement, reconstruction and scenarios support different uses',17,bold=True,anchor='middle')
text(400,277,'Revised SUEIHCDR model  |  Three California counties  |  Retrospective evaluation',12,anchor='middle',color=GRAY)
C=['San Diego','Los Angeles','San Francisco'];labels=['SD','LA','SF']
root=B/'data'
s=pd.read_csv(root/'serology_residual_summary.csv');r=pd.read_csv(root/'reconstruction.csv');f=pd.read_csv(root/'future_week_summary.csv')
def axes(x,title,ymax,ticks):
 text(x+110,244,title,13,bold=True,anchor='middle')
 for val in ticks:
  yy=86+val/ymax*112;line(x,yy,x+211,yy,LIGHT);text(x-7,yy-3,f'{val:g}',10,anchor='end',color=GRAY)
 line(x,86,x+211,86);line(x,86,x,198)
 for i,label in enumerate(labels):text(x+36+i*69,66,label,12,anchor='middle')
 return lambda val:val/ymax*112
scale=axes(43,'Infection scale agreement',.5,[0,.1,.2,.3,.4,.5])
text(148,222,'Mean absolute donor difference',11,anchor='middle',color=GRAY)
for i,c in enumerate(C):
 for model,shift,color in [('L3',-13,GRAY),('L3B',3,BLUE)]:
  value=float(s[(s.county==c)&(s.period=='All dates')&(s.model==model)].iloc[0].mean_abs_diff)
  rect(43+36+i*69+shift,86,12,scale(value),color)
rect(50,39,9,9,GRAY);text(64,39,'Before',10);rect(124,39,9,9,BLUE);text(138,39,'After serology calibration',10)
scale=axes(311,'Continuous reconstruction',2.2,[0,.5,1,1.5,2])
text(416,222,'Model / observed admissions',11,anchor='middle',color=GRAY)
line(311,86+scale(1),522,86+scale(1),BLACK,[3,2])
for i,c in enumerate(C):
 for period,shift,color in [('behaviour',-13,ORANGE),('vaccination',3,BLUE)]:
  val=float(r[(r.county==c)&(r.period==period)].iloc[0].ratio)
  rect(311+36+i*69+shift,86,12,scale(val),color)
rect(317,39,9,9,ORANGE);text(331,39,'October start',10);rect(411,39,9,9,BLUE);text(425,39,'December start',10)
scale=axes(579,'Five scenario bracketing',100,[0,25,50,75,100])
text(684,222,'First four wholly future weeks (%)',11,anchor='middle',color=GRAY)
for i,c in enumerate(C):
 a=f[(f.county==c)&(f.model=='L3B')&(f.delay==1)&(f.window=='future1_4')].iloc[0]
 v=100*a.coverage;rect(579+36+i*69-10,86,20,scale(v),BLUE)
 text(579+36+i*69,93+scale(v),f'{v:.1f}%',11,bold=True,anchor='middle')
text(684,39,'Observed admission targets bracketed',10,anchor='middle')
text(400,9,'Selected scenario paths are not probability intervals. Vaccination contrasts remain conditional.',12,anchor='middle')
out=P/'Graphical_abstract_Epidemics';renderSVG.drawToFile(d,str(out.with_suffix('.svg')));renderPDF.drawToFile(d,str(out.with_suffix('.pdf')))
pdf=pdfium.PdfDocument(out.with_suffix('.pdf'));im=pdf[0].render(scale=2).to_pil();im.save(out.with_suffix('.png'),dpi=(144,144))
(P/'Graphical_abstract_caption.txt').write_text('Graphical abstract\nPrepared as vector data graphics with ReportLab, using the archived numerical tables and the script prepare_graphics.py. No generative image tool was used. SD: San Diego; LA: Los Angeles; SF: San Francisco. Mean absolute donor differences use the same 21 dates for each model and county. Reconstruction ratios use comparable observed weeks; the October and December periods have different states and inputs. Scenario bracketing uses the first four wholly future calendar weeks and has no nominal probability level.\n',encoding='utf-8')
print('Prepared graphical abstract',im.size)
