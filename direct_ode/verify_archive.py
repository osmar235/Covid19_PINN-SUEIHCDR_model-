from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
m=json.loads((R/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
bad=[]
for row in m['files']:
 p=R/row['path'];h=hashlib.sha256()
 if not p.is_file():bad.append(row['path']);continue
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 if h.hexdigest()!=row['sha256']:bad.append(row['path'])
if bad:raise SystemExit('Hash mismatches: '+str(bad))
print('PASS: '+str(len(m['files']))+' archive files verified')
