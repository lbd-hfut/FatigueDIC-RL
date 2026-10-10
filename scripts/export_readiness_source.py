"""Verified source-only incremental export into a new directory; no Git writes."""
import argparse,difflib,hashlib,json,subprocess,tarfile,tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(run,output):
    run,output=Path(run),Path(output);output.mkdir(parents=True,exist_ok=False)
    before=json.loads((run/'baseline/source_manifest.json').read_text())
    paths=set(before)
    for directory in ('src','scripts','tests','configs','docs','.github'):
        paths.update(str(p) for p in Path(directory).rglob('*') if p.is_file()
            and p.suffix in ('.py','.md','.toml','.json','.yaml','.yml') and '__pycache__' not in p.parts)
    manifest={p:digest(p) for p in sorted(paths)}
    assert all(manifest[p]==h for p,h in before.items()),'historical 327 source files changed'
    (output/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    with tarfile.open(output/'source_final.tar.gz','w:gz') as archive:
        for path in manifest:archive.add(path,arcname=path,recursive=False)
    original={}
    with tarfile.open(run/'baseline/source_predevelopment.tar.gz') as archive:
        for m in archive.getmembers():
            if m.isfile():original[m.name]=archive.extractfile(m).read().decode()
    patch=''.join(''.join(difflib.unified_diff(original.get(p,'').splitlines(True),Path(p).read_text().splitlines(True),
        fromfile='a/'+p if p in original else '/dev/null',tofile='b/'+p)) for p in manifest)
    (output/'incremental.patch').write_text(patch)
    restored={}
    for kind in ('archive','patch'):
        dest=Path(tempfile.mkdtemp(prefix='fatigue-readiness-final-'+kind+'-',dir='/tmp'))
        source=output/'source_final.tar.gz' if kind=='archive' else run/'baseline/source_predevelopment.tar.gz'
        with tarfile.open(source) as archive:archive.extractall(dest,filter='data')
        if kind=='patch':
            r=subprocess.run(['patch','--batch','-p1','--input',str((output/'incremental.patch').resolve())],cwd=dest,capture_output=True,text=True)
            (output/'patch_restoration.log').write_text(r.stdout+r.stderr);assert r.returncode==0
        assert {str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()}==set(manifest)
        assert all(digest(dest/p)==h for p,h in manifest.items());restored[kind]=str(dest)
    audit=dict(source_files=len(manifest),source_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        manifest_sha256=digest(output/'manifest.json'),archive_sha256=digest(output/'source_final.tar.gz'),
        patch_sha256=digest(output/'incremental.patch'),restored=restored,restoration_all_files_verified=True,
        historical_327_files_unchanged=True,git_writes=False)
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (output/'RECOVERY.md').write_text('Use a new empty directory. Extract source_final.tar.gz and verify every file with manifest.json. Alternatively extract ../baseline/source_predevelopment.tar.gz (the verified Phase 2F archive) and apply this incremental.patch using patch -p1. Both routes were actually verified: '+json.dumps(restored)+'. This source package excludes results, checkpoints, archives, caches and Git internals. Full scientific assets require the separate delivery ledger and original run directories.\n')
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(export(args.run,args.output)))
