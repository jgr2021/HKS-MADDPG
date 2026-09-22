"""Catalogue mapping and explicit, isolated preparation. Default never trains."""
import argparse
import csv
import ctypes
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIVE = ROOT / 'experiments/gsp_long_20260906b'
CATALOG = ROOT / 'docs/maddpg_gsp_design_catalog_20260906.md'
ALIASES = {
    'G1':'explore_hks_3aa', 'G2':'explore_hks_4ego', 'G3':'explore_hks_6al',
    'G4':'explore_hks_6aal', 'G5':'explore_hks_6all',
    'E1':'explore_hks_6al', 'E3':'explore_hks_knn2_union', 'E5':'explore_hks_soft_radius',
    'W2':'explore_hks_6al', 'W3':'explore_hks_inverse',
    'W6':'explore_hks_sigma075', 'O2':'explore_hks_6al', 'O3':'explore_hks_combinatorial',
    'O4':'explore_rwse_248', 'O5':'explore_hks_self_loop', 'O6':'explore_lazy_rwse_248',
    'X1':'explore_hks_6al', 'D1':'explore_hks_6al',
    'D3':'explore_wks3', 'D6':'explore_landmark_heat3',
    'A1':'explore_hks_6al', 'A3':'explore_contact_d4_energy', 'A7':'explore_contact_d4_energy',
    'R1':'explore_hks_6al', 'R8':'explore_contact_d4_energy',
}
NEW = {
    'G6':'split_hks','G7':'pair_hks','G8':'hypergraph_hks',
    'E2':'radius_hks','E4':'mutual_hks','E6':'landmark_top2_hks','E7':'weak_long_hks','E8':'encoder_directed',
    'W1':'binary_hks','W4':'compact_hks','W5':'relation_scales_hks','W7':'adaptive_hks','W8':'encoder_learned','W9':'contact_weight_hks','W10':'encoder_directed',
    'O1':'adjacency_readout','O7':'split_hks','O8':'coordinate_filter',
    'X2':'coordinate_filter','X3':'coverage_filter','X4':'crowding_filter','X5':'affinity_filter',
    'X6':'advantage_filter','X7':'velocity_filter','X8':'degree_stats','X9':'mask_filter',
    'D2':'rwse246','D4':'effective_resistance','D5':'spectrum_moments','D7':'coordinate_filter','D8':'static_energy',
    'D9':'rayleigh_energy','D10':'cross_energy','D11':'wavelet_signal',
    'A2':'action_absolute','A4':'action_current_delta','A6':'action_self_contact',
    'A8':'action_hypotheses','A9':'action_horizon3',
    'R3':'split_hks','R4':'encoder_sets','R5':'encoder_attention','R6':'encoder_sets','R7':'encoder_gcn',
    'R9':'vector_projection','R10':'layernorm_hks',
}
LEGACY = {
    'A5':('d4_dirichlet_residual','d4_potential_residual'),
    'R2':('local_spectral_al_residual','local_geometry_residual'),
    'R11':('local_spectral_al_residual','local_geometry_residual'),
}
BLOCKED = {
    'D12':'Raw eigenvectors/Fourier coordinates need a signed/repeated-eigenspace policy and matched control; no silent substitute with invariant descriptors.',
    'R12':'GRU requires sequence replay, burn-in, hidden-state reset and a separate training protocol; flat replay must not be silently reused.',
}
DEFAULTS = dict(total_env_steps=2500000, checkpoint_interval_env_steps=20000,
                eval_episodes=500, curve_eval_episodes=100, seed=1, batch_size=1024,
                n_rollout_threads=4, max_workers=2, smoke_env_steps=2000)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepared_source_campaign(manifest):
    """Return the immutable source campaign recorded when this template was made."""
    source = manifest.get('prepared_from')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Prepared catalogue manifest is missing its source campaign path')
    source_path = Path(source).resolve()
    if not source_path.is_dir():
        raise ValueError('Prepared catalogue source campaign is unavailable: ' + str(source_path))
    return source_path


def memory_info():
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
            ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extra')]
    s=Status()
    s.length=ctypes.sizeof(s)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
        raise OSError('Cannot inspect memory headroom')
    return dict(avail_phys=s.avail_phys,avail_page=s.avail_page)


def register():
    from experiments.run_gsp_long_campaign import REGISTRATIONS
    for module in REGISTRATIONS:
        importlib.import_module(module).register_policies()
    from utils.catalog76_policies import register_policies
    register_policies()


def entries():
    rows = []
    for line in CATALOG.read_text(encoding='utf-8').splitlines():
        match = re.match(r'^\| ([GEWOXDAR]\d+) \| (.*) \|$', line)
        if match:
            parts = match[2].split(' | ')
            key = match[1]
            actor = ALIASES.get(key) or ('catalog_'+NEW[key] if key in NEW else LEGACY.get(key, (None,))[0])
            rows.append(dict(id=key, design=parts[0], rationale=parts[1],
                original_caveat=' | '.join(parts[2:]), actor_model=actor,
                status='blocked_protocol' if key in BLOCKED else 'implemented_pending_training_validation',
                blocker=BLOCKED.get(key), recipe=NEW.get(key),
                mapping_scope='A concrete representative, not all variants or Cartesian combinations',
                existing_alias=key in ALIASES or key in LEGACY))
    expected = set(ALIASES)|set(NEW)|set(LEGACY)|set(BLOCKED)
    assert len(rows)==76 and {r['id'] for r in rows}==expected
    return rows


def selected_methods(ids):
    register()
    from utils.agents import POLICY_TYPES
    from experiments.run_gsp_long_campaign import methods as existing_methods
    from utils.catalog76_policies import FEATURE_RECIPES
    existing = existing_methods()
    by_actor = {v['actor_model']:(k,v) for k,v in existing.items()}
    output = {}
    feature_controls = {(21, False): 'explore_geometric_stats3'}

    def include(name, spec):
        if name in output:
            return
        for control in [spec.get('control')] + spec.get('required_additional_controls', []):
            if control and control not in output:
                include(control, existing[control])
        output[name] = dict(spec)

    include('raw_mlp', existing['raw_mlp'])
    lookup = {r['id']:r for r in entries()}
    for key in ids:
        if key not in lookup:
            raise ValueError('Unknown catalogue ID: '+key)
        entry = lookup[key]
        if entry['blocker']:
            raise ValueError(key+': '+entry['blocker'])
        actor = entry['actor_model']
        if actor in by_actor:
            name,spec = by_actor[actor]
            include(name,spec)
            if key=='W6':
                for extra in ('explore_hks_6al','explore_hks_sigma125'):
                    include(extra,existing[extra])
            continue
        if actor not in POLICY_TYPES:
            raise ValueError('Unregistered actor: '+actor)
        model = POLICY_TYPES[actor](18,5)
        dim = getattr(model,'actor_input_dim',18)
        control = LEGACY[key][1] if key in LEGACY else actor+'_control'
        if entry.get('recipe') in FEATURE_RECIPES and FEATURE_RECIPES[entry['recipe']].get('descriptor') != 'action':
            control_key=(dim,bool(FEATURE_RECIPES[entry['recipe']].get('layernorm')))
            control=feature_controls.setdefault(control_key,control)
        control_spec=existing.get(control,dict(actor_model=control,actor_input_dim=dim,control='raw_mlp',group='catalog_control'))
        include(control,control_spec)
        config = FEATURE_RECIPES.get(entry.get('recipe'), {'encoder':entry.get('recipe')})
        include(actor,dict(actor_model=actor, actor_input_dim=dim, control=control,group='catalog',
                           catalog_ids=[r['id'] for r in lookup.values() if r['actor_model']==actor],
                           recipe=config, factor=entry['design']))
    for index,(name,spec) in enumerate(output.items()):
        spec.update(short='c%02d'%index, fresh=True)
        spec.pop('reuse_from',None)
    return output


def export_catalog(destination):
    destination.mkdir(parents=True, exist_ok=True)
    payload=dict(generated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),defaults=DEFAULTS,entries=entries(),
                 training_started=False, scope='76 design options, not 76 independent runs')
    (destination/'catalog.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    with (destination/'catalog.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(payload['entries'][0]))
        writer.writeheader()
        writer.writerows(payload['entries'])
    print(json.dumps({'entries':76,'blocked':list(BLOCKED),'training_started':False}))


def prepare(destination, ids):
    destination=destination.resolve()
    if destination.exists():
        raise FileExistsError('No overwrite or resume: '+str(destination))
    # Keep future frozen campaigns out of any current campaign or its code subtree.
    if LIVE.resolve()==destination or LIVE.resolve() in destination.parents:
        raise ValueError('Cannot prepare inside the live campaign')
    if any(p.name.startswith('gsp_long_') for p in destination.parents if p!=ROOT):
        raise ValueError('Choose an isolated output outside previous campaigns')
    methods=selected_methods(ids)
    from experiments.run_gsp_long_campaign import training_args
    from experiments.run_gsp_exploration_campaign import code_hashes, verify_snapshot
    verify_snapshot(LIVE)
    if shutil.disk_usage(destination.parent).free < 8*1024**3:
        raise RuntimeError('At least 8 GiB free required')
    destination.mkdir(parents=True)
    # Reuse the proven protocol source, never the mutable training checkout.
    shutil.copytree(LIVE/'code',destination/'code')
    for relative in ('utils/catalog76_policies.py','experiments/catalog76.py'):
        shutil.copy2(ROOT/relative,destination/'code'/relative)
    manifest=read(LIVE/'manifest.json')
    manifest.update(methods=methods,source_hashes=code_hashes(destination/'code'),catalog_ids=ids,
                    study='catalog76_prepared_not_started',formal=vars(training_args('formal')),
                    smoke=vars(training_args('smoke')),prepared_from=str(LIVE),
                    queue_rule='Explicit --execute required; at most two workers, smoke before formal; stop admissions on failure',
                    inherited_completed_methods=[],resume=False,reuse_old_results=False)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (destination/'catalog_selection.json').write_text(json.dumps(dict(ids=ids,methods=list(methods),
        catalog_source_sha256=hashlib.sha256(CATALOG.read_bytes()).hexdigest(),training_started=False),indent=2),encoding='utf-8')
    print(json.dumps({'prepared':str(destination),'methods':list(methods),'training_started':False}))


def run(destination, execute, inherited_done=(), on_complete=None, continuation=None,
        priority_methods=(), adopted_active=None, urgent_methods=()):
    if not execute:
        raise ValueError('Training is disabled without explicit --execute')
    from experiments.run_gsp_exploration_campaign import verify_snapshot
    manifest=verify_snapshot(destination)
    if len(set(priority_methods)) != len(priority_methods) or not set(priority_methods) <= set(manifest['methods']):
        raise ValueError('Invalid priority methods')
    dispatch_order=list(priority_methods)+[n for n in manifest['methods'] if n not in priority_methods]
    source_campaign = prepared_source_campaign(manifest)
    if destination.resolve()==source_campaign:
        raise ValueError('Never run the catalogue controller against the live campaign')
    if (source_campaign/'parallel_status.json').exists():
        live_status=read(source_campaign/'parallel_status.json')
        if live_status.get('active'):
            raise RuntimeError('Existing campaign still has active workers; do not add another training queue')
    if continuation is None:
        with (destination/'catalog_started.json').open('x',encoding='utf-8') as stream:
            json.dump({'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'resume':False},stream)
    active,done,smoke,failures={},set(inherited_done),set(),[]
    active.update(adopted_active or {})
    if not set(urgent_methods) <= set(priority_methods):
        raise ValueError('Urgent methods must have explicit priority')
    skipped=set()
    if continuation is not None:
        done.update(continuation['completed'])
        smoke.update(continuation['smoke_completed'])
        skipped.update(continuation['skipped'])
        if done & skipped or not skipped <= set(manifest['methods']):
            raise ValueError('Invalid continuation inventory')
    try:
        while not set(manifest['methods']) <= done | skipped:
            for name,item in list(active.items()):
                code=item['process'].poll()
                if code is None:
                    continue
                del active[name]
                marker=destination/(item['phase']+'_'+name+'.json')
                state=read(marker) if marker.exists() else {}
                if code!=0 or state.get('status')!='completed' or not state.get('validation',{}).get('passed'):
                    failures.append(dict(method=name,phase=item['phase'],exit_code=code,error=state.get('error')))
                else:
                    (done if item['phase']=='formal' else smoke).add(name)
                    if item['phase']=='formal' and on_complete is not None:
                        try:
                            on_complete(name)
                        except Exception as error:
                            failures.append(dict(method=name, phase='analysis', error=repr(error)))
            paused=(destination/'PAUSE_AFTER_CURRENT').exists() or bool(failures)
            for name in dispatch_order:
                spec=manifest['methods'][name]
                if paused:
                    break
                if len(active)>=2 and not (name in urgent_methods and len(active)<3):
                    continue
                if name in active or name in done | skipped:
                    continue
                controls=[spec.get('control')]+spec.get('required_additional_controls',[])
                if name not in urgent_methods and any(c and c not in done for c in controls):
                    continue
                mem=memory_info()
                if mem['avail_phys']<3*1024**3 or mem['avail_page']<6*1024**3:
                    break
                if shutil.disk_usage(destination).free<2*1024**3:
                    failures.append(dict(error='Insufficient disk headroom; no deletion'))
                    break
                phase='formal' if name in smoke else 'smoke'
                if (destination/phase/name/'seed_1').exists():
                    raise FileExistsError('Refusing to overwrite existing attempt: '+name)
                verify_snapshot(destination)
                command=[sys.executable,'-u','-B',str(destination/'code/experiments/catalog76.py'),
                         'worker','--campaign',str(destination),'--method',name,'--phase',phase,'--execute']
                with (destination/(phase+'_'+name+'.process.log')).open('x',encoding='utf-8') as log:
                    process=subprocess.Popen(command,cwd=destination,stdout=log,stderr=subprocess.STDOUT,
                                             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                active[name]=dict(process=process,phase=phase)
            state=dict(status='draining_after_failure' if failures and active else
                       'paused_technical_failure' if failures else 'paused_by_user' if paused else
                       'completed_with_skips' if skipped and set(manifest['methods']) <= done | skipped else
                       'completed' if set(manifest['methods']) <= done else 'running',
                       completed=sorted(set(manifest['methods']) & done),inherited_completed=sorted(set(inherited_done)),failures=failures,
                       skipped=sorted(skipped),
                       active=[dict(method=n,phase=i['phase'],pid=i['process'].pid) for n,i in active.items()],
                       checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
            tmp=destination/'catalog_status.tmp'
            tmp.write_text(json.dumps(state,indent=2),encoding='utf-8')
            tmp.replace(destination/'catalog_status.json')
            if (paused or failures) and not active:
                return
            if not set(manifest['methods']) <= done | skipped:
                time.sleep(10)
    except BaseException as error:
        (destination/'catalog_controller_error.json').write_text(json.dumps(dict(error=repr(error),
            active=[dict(method=n,phase=i['phase'],pid=i['process'].pid) for n,i in active.items()])),encoding='utf-8')
        raise


def worker(destination, method, phase, execute):
    if not execute or ROOT!=destination/'code':
        raise ValueError('Worker requires explicit execution of its frozen snapshot')
    register()
    from experiments import run_gsp_long_campaign as c
    c.worker(destination,phase,method)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('list','prepare','run','worker'))
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/catalog76_preparation')
    parser.add_argument('--ids',nargs='+')
    parser.add_argument('--campaign',type=Path)
    parser.add_argument('--method')
    parser.add_argument('--phase',choices=('smoke','formal'))
    parser.add_argument('--execute',action='store_true')
    options=parser.parse_args()
    if options.command=='list':
        export_catalog(options.output)
    elif options.command=='prepare':
        if not options.ids:
            parser.error('Specify catalogue IDs; no implicit all-method launch')
        prepare(options.output,options.ids)
    elif options.command=='run':
        if not options.campaign:
            parser.error('--campaign required')
        run(options.campaign.resolve(),options.execute)
    else:
        if not all((options.campaign,options.method,options.phase)):
            parser.error('--campaign, --method and --phase required')
        worker(options.campaign.resolve(),options.method,options.phase,options.execute)


if __name__=='__main__':
    main()
