"""Recoverable source-only archive and verified incremental patch, without Git writes."""
import argparse,difflib,hashlib,json,subprocess,tarfile,tempfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    paths={Path(p) for p in json.loads((root/'baseline/source_manifest.json').read_text())}
    for directory in ('src','tests','scripts','configs','docs','.github'):
        paths.update(p for p in Path(directory).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.toml','.yaml','.yml','.json') and '__pycache__' not in p.parts)
    paths=sorted(p for p in paths if p.is_file())
    manifest={str(p):sha(p) for p in paths}
    canonical=json.dumps(manifest,sort_keys=True,separators=(',',':'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    with tarfile.open(out/'source_final.tar.gz','w:gz') as archive:
        for path in paths:archive.add(path,arcname=str(path),recursive=False)
    before={}
    with tarfile.open(root/'baseline/source_predevelopment.tar.gz') as archive:
        for m in archive.getmembers():
            if m.isfile():before[m.name]=archive.extractfile(m).read().decode()
    patch=''.join(''.join(difflib.unified_diff(before.get(str(p),'').splitlines(True),p.read_text().splitlines(True),
        fromfile='a/'+str(p) if str(p) in before else '/dev/null',tofile='b/'+str(p))) for p in paths)
    (out/'incremental.patch').write_text(patch)
    restoration=Path(tempfile.mkdtemp(prefix='fatigue-phase2c-restoration-',dir='/tmp'))
    with tarfile.open(root/'baseline/source_predevelopment.tar.gz') as archive:archive.extractall(restoration,filter='data')
    applied=subprocess.run(['patch','--batch','-p1','--input',str((out/'incremental.patch').resolve())],cwd=restoration,capture_output=True,text=True)
    (out/'patch_restoration.log').write_text(applied.stdout+applied.stderr)
    if applied.returncode:raise RuntimeError('incremental patch restoration failed')
    reconstructed={p:sha(restoration/p) for p in manifest}
    assert reconstructed==manifest
    audit={'source_files':len(manifest),'source_sha256':hashlib.sha256(canonical.encode()).hexdigest(),
        'hash_definition':'SHA256 of sorted path-to-SHA256 mapping serialized as canonical JSON separators comma/colon',
        'archive_sha256':sha(out/'source_final.tar.gz'),'manifest_sha256':sha(out/'manifest.json'),'incremental_patch_sha256':sha(out/'incremental.patch'),
        'restoration_directory':str(restoration),'patch_restoration_matches_all_final_files':True,
        'excluded':'all results, old tar files, checkpoints, caches, Git internals','git_writes':False}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (out/'RECOVERY.md').write_text('Restore only into a **new empty directory**. Extract `../baseline/source_predevelopment.tar.gz` there, then apply `incremental.patch` with `patch -p1`. Alternatively extract `source_final.tar.gz`. Verify every path with `manifest.json`. The incremental restoration was actually executed in '+str(restoration)+' and matched every final file. No Git reset/clean or historic result replacement is needed. Research result assets remain separate; source archives never recursively include results.\n')
    (out/'git_status.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    (out/'git_HEAD.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'],text=True))
    # Hash new research assets only after all computations and report generation finish.
    assets={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='delivery_assets.json'}
    (root/'delivery_assets.json').write_text(json.dumps(assets,indent=2,sort_keys=True)+'\n')
    print(json.dumps(audit),flush=True)
