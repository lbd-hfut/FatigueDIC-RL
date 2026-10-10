"""Frozen case-level dataset and five-seed low-dimensional residual learning."""
import argparse,json,tomllib,hashlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from fatigue_dic_rl.training.correction_dataset import generate_correction_case,source_hash


def generate_task(task):
    folder,kwargs=task;return generate_correction_case(folder,**kwargs)


def training_task(task):
    train,val,out,kwargs=task
    from fatigue_dic_rl.training.correction_supervised import train_residual_coefficients
    return train_residual_coefficients(train,val,out,**kwargs)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/phase2c/study.toml');ap.add_argument('--stage',choices=['dataset','supervised'],required=True);ap.add_argument('--dataset');ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);text=Path(a.config).read_text();config=tomllib.loads(text);cfgsha=hashlib.sha256(text.encode()).hexdigest();(out/'config.toml').write_text(text);t=time.perf_counter()
    fixed={'A_prior':8e-5,'m_prior':1.,'Y_ref':.025,'future_load_factor':.25,'assimilation_cycles':[0.,10.,20.],'forecast_cycles':[30.,40.,50.,70.,120.,220.]}
    if any(config['physics'][k]!=v for k,v in fixed.items()):
        raise ValueError('This prototype supports only its declared nominal law and time split; unsupported config changes cannot be ignored')
    if config['training']['learning_rate']!=.02:
        raise ValueError('Residual coefficient trainer currently declares fixed AdamW learning rate .02')
    if a.stage=='dataset':
        tasks=[]
        for regime in config['dataset']['regimes']:
            for split in ('train','validation','calibration','test'):
                for index in range(config['dataset'][split]):
                    seed=config['dataset'][split+'_first_seed']+index
                    tasks.append((str(out/regime/split/f'case_{seed}'),dict(seed=seed,split=split,scenario=regime,sigma_mm=config['dataset']['sigma_mm'],scales=tuple(config['physics']['load_scales']),step=config['physics']['cycle_step'],R=config['physics']['stress_ratio'],beta=config['physics']['beta_structural'],config_sha256=cfgsha)))
        with ProcessPoolExecutor(max_workers=config['dataset']['workers']) as pool:rows=list(pool.map(generate_task,tasks))
        manifest={'records':rows,'source_sha256':source_hash(),'config_sha256':cfgsha,'split_unit':'entire material-case group; five public load specimens; no frames across splits','paired_regime_seeds':True,'future_training_fields_generated':False,'wall_seconds':time.perf_counter()-t}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('generated',len(rows),'groups',flush=True)
    else:
        records=json.loads((Path(a.dataset)/'manifest.json').read_text())['records'];tasks=[]
        for regime in config['dataset']['regimes']:
            train=[r['path'] for r in records if r['scenario']==regime and r['split']=='train'];val=[r['path'] for r in records if r['scenario']==regime and r['split']=='validation']
            for k in config['training']['dimensions']:
                for seed in config['training']['seeds']:
                    tasks.append((train,val,str(out/regime/f'K{k}'/f'seed_{seed}'),dict(seed=seed,dimension=k,alpha=config['training']['alpha'],epochs=config['training']['maximum_epochs'],patience=config['training']['patience'])))
        with ProcessPoolExecutor(max_workers=config['dataset']['workers']) as pool:rows=list(pool.map(training_task,tasks))
        (out/'manifest.json').write_text(json.dumps({'models':rows,'source_sha256':source_hash(),'config_sha256':cfgsha,'wall_seconds':time.perf_counter()-t},indent=2)+'\n');print('trained',len(rows),'checkpoints',flush=True)
