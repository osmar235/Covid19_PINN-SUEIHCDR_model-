"""Retrieve the single large HHS snapshot omitted from the Git source tree.

The full release ZIP already contains it and needs no network access. This
helper uses only Python's standard library and verifies the frozen input hash.
"""
from pathlib import Path
import hashlib,json,shutil,tempfile,urllib.request,zipfile

ROOT=Path(__file__).resolve().parent
REL='data_epi/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_Facility.csv'
URL='https://github.com/osmar235/Covid19_PINN-SUEIHCDR_model-/releases/download/ode-v1.2.0/COVID19_ODE_reproducibility_v1.2.0.zip'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def main():
    inventory=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    expected=next(row for row in inventory['files'] if row['path']==REL)
    target=ROOT/REL
    if target.exists():
        if sha(target)!=expected['sha256']:raise SystemExit('Existing HHS input has a different hash; no file was overwritten.')
        print('Frozen HHS input already present; SHA-256 verified.');return
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hhs_snapshot_',dir=ROOT) as temporary:
        temp=Path(temporary);archive=temp/'snapshot.zip'
        print('Downloading the versioned release archive:',URL,flush=True)
        req=urllib.request.Request(URL,headers={'User-Agent':'COVID19-ODE-reproducibility'})
        with urllib.request.urlopen(req,timeout=180) as response, archive.open('wb') as dest:
            shutil.copyfileobj(response,dest,1<<20)
        frozen=temp/'hhs.csv'
        with zipfile.ZipFile(archive) as z, z.open('covid19_ode_repro_v1.2.0/'+REL) as source, frozen.open('wb') as dest:
            shutil.copyfileobj(source,dest,1<<20)
        if frozen.stat().st_size!=expected['bytes'] or sha(frozen)!=expected['sha256']:
            raise SystemExit('Downloaded HHS input failed the frozen checksum; no input was installed.')
        if target.exists():raise SystemExit('An HHS input appeared during download; no existing file was overwritten.')
        shutil.move(str(frozen),str(target))
    print('Installed frozen HHS input; SHA-256 verified. Run python verify_archive.py next.')
if __name__=='__main__':main()
