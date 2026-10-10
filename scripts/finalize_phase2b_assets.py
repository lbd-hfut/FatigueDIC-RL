"""Preserve immutable evidence and verify source/patch recovery; no Git writes."""
import argparse,hashlib,json,tarfile,tempfile,difflib,subprocess
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root);out=root/'delivery';out.mkdir(parents=True,exist_ok=False);before=json.loads((root/'baseline/source_manifest.json').read_text());old_results=json.loads((root/'baseline/existing_results_manifest.json').read_text());changed_results=[p for p,h in old_results.items() if not Path(p).is_file() or sha(p)!=h]
    protected=[p for p in before if p.startswith(('tests/','configs/phase0','configs/phase1','configs/phase2a','src/fatigue_dic_rl/mechanics/','src/fatigue_dic_rl/fatigue/','src/fatigue_dic_rl/observations/','src/fatigue_dic_rl/inverse/')) or p.startswith('docs/PHASE')];changed_protected=[p for p in protected if sha(p)!=before[p]]
    assert not changed_results and not changed_protected,(changed_results,changed_protected)
    files=sorted({p for folder in ['src','scripts','tests','configs','docs','.github'] for p in Path(folder).rglob('*') if p.is_file() and p.suffix not in ('.pyc',) and '__pycache__' not in p.parts}|{Path(p) for p in ['README.md','LICENSE','pyproject.toml','.gitignore']});after={str(p):sha(p) for p in files};manifest=out/'source_manifest.json';manifest.write_text(json.dumps(after,indent=2)+'\n')
    aggregate=hashlib.sha256('\n'.join(f'{p}:{h}' for p,h in after.items()).encode()).hexdigest()
    restore=Path(tempfile.mkdtemp(prefix='fatigue-2b-recovery-'))
    with tarfile.open(root/'baseline/source_before_phase2b.tar.gz') as tar:tar.extractall(restore,filter='data')
    patch=[]
    for name in sorted(set(before)|set(after)):
        old=(restore/name).read_text() if name in before else '';new=Path(name).read_text() if name in after else ''
        if old!=new:patch.extend(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
    patch_path=out/'increment_from_predevelopment.patch';patch_path.write_text(''.join(patch))
    with patch_path.open() as stdin:r=subprocess.run(['patch','-p1','--batch'],cwd=restore,stdin=stdin,capture_output=True,text=True)
    (out/'patch_recovery.log').write_text(r.stdout+r.stderr);recovered=[name for name,h in after.items() if (restore/name).is_file() and sha(restore/name)==h];assert r.returncode==0 and len(recovered)==len(after)
    archive=out/'source_final.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p),recursive=False)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==after[member.name]
    evidence={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and 'delivery' not in p.relative_to(root).parts};(out/'results_manifest.json').write_text(json.dumps(evidence,indent=2)+'\n')
    result={'source_aggregate_sha256':aggregate,'source_file_count':len(after),'source_manifest_sha256':sha(manifest),'final_tar_sha256':sha(archive),'increment_patch_sha256':sha(patch_path),'results_manifest_sha256':sha(out/'results_manifest.json'),'result_file_count':len(evidence),'prior_result_count':len(old_results),'prior_results_unchanged':not changed_results,'protected_original_source_test_config_report_count':len(protected),'protected_originals_unchanged':not changed_protected,'recovery_file_count':len(recovered),'recovery_directory':str(restore),'git_writes':False,'snapshots_exclude_results_and_old_archives':True};(out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
