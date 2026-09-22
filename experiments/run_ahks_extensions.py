"""MAPPO, heuristic evaluation and complete extension campaign entry point."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from algorithms.ahks_mappo import MAPPO, generalized_advantage, ppo_update
from utils.ahks_environments import make_task, navigation_controller, sensing_controller, SensingConfig


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def csv_write(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    paths = [Path(__file__), ROOT/'algorithms/ahks_mappo.py', ROOT/'utils/ahks_environments.py',
             ROOT/'utils/catalog76_policies.py', ROOT/'utils/exploration_topology_policies.py',
             ROOT/'utils/make_env.py']
    import multiagent
    paths += sorted(Path(multiagent.__file__).parent.rglob('*.py'))
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def evaluate(task, episodes, seed_start, model=None, controller=None, device='cpu', config=None):
    env = make_task(task, seed_start, config)
    rows = []
    try:
        for episode in range(episodes):
            obs = env.reset(seed_start+episode)
            done = False
            while not done:
                if model is not None:
                    actions = model.act(torch.as_tensor(obs[None], device=device), deterministic=True)[0][0].cpu().numpy()
                elif task == 'spread':
                    actions = navigation_controller(obs, controller)
                else:
                    actions = sensing_controller(obs, controller, config)
                obs, _, done, info = env.step(actions)
            rows.append(dict(test_seed=seed_start+episode, **info['metrics']))
    finally:
        env.close()
    return rows


def summarize(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != 'test_seed'}


def save_checkpoint(path, model, optimizer, config, steps):
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                    steps=steps, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state()), path)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    model = MAPPO(cfg['obs_dim'], cfg['descriptor'], cfg['hidden']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


def train(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite ' + str(output))
    if min(args.steps, args.envs, args.rollout, args.epochs, args.minibatch,
           args.checkpoint_interval, args.eval_episodes, args.curve_episodes, args.hidden) <= 0:
        raise ValueError('Budgets and dimensions must be positive')
    if args.steps % args.envs or args.checkpoint_interval % args.envs:
        raise ValueError('Budgets/checkpoint intervals must be divisible by env count')
    hashes = source_hashes()
    if args.phase == 'formal':
        if args.smoke_reference is None:
            raise ValueError('Formal training requires --smoke-reference validation.json')
        smoke = json.loads(args.smoke_reference.read_text())
        if not smoke.get('passed') or smoke['task'] != args.task or smoke['descriptor'] != args.descriptor or smoke['source_hashes'] != hashes:
            raise ValueError('Smoke validation does not match this task/descriptor/source')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    envs = [make_task(args.task, args.seed*10000+i) for i in range(args.envs)]
    model = MAPPO(envs[0].obs_dim, args.descriptor, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(obs_dim=envs[0].obs_dim, device=device, action_mapping=['stay', '+x', '-x', '+y', '-y'],
                  actor_input_dim=model.actors[0].input_dim, critic_input_dim=3*envs[0].obs_dim,
                  sensing_config=asdict(SensingConfig()) if args.task == 'sensing' else None,
                  source_hashes=hashes, versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
                  implementation='local feed-forward MAPPO, independent actors, team-value critic',
                  truncation='Spread time-limit bootstraps pre-reset value; traces stop. Sensing horizon is terminal.')
    output.mkdir(parents=True)
    write(output/'protocol.json', config)
    checkpoints = output/'checkpoints'
    checkpoints.mkdir()
    initial_actor = [next(a.parameters()).detach().clone() for a in model.actors]
    initial_critic = next(model.critic.parameters()).detach().clone()
    obs = np.stack([env.reset() for env in envs])
    steps = updates = 0
    curve, training_rows = [], []
    started = time.perf_counter()
    last_batch = None
    try:
        while steps < args.steps:
            length = min(args.rollout, (args.steps-steps)//args.envs,
                         (args.checkpoint_interval-steps % args.checkpoint_interval)//args.envs)
            storage = [[] for _ in range(8)]
            for _ in range(length):
                tensor = torch.as_tensor(obs, device=device)
                with torch.no_grad():
                    actions, logp, value = model.act(tensor)
                transitions = [env.step(action) for env, action in zip(envs, actions.cpu().numpy())]
                next_obs = np.stack([item[0] for item in transitions])
                with torch.no_grad():
                    next_value = model.value(torch.as_tensor(next_obs, device=device)).cpu().numpy()
                data = [obs.copy(), actions.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy(),
                        np.array([item[1] for item in transitions], dtype=np.float32), next_value,
                        np.array([item[3]['terminated'] for item in transitions], dtype=np.float32),
                        np.array([item[2] for item in transitions], dtype=np.float32)]
                for stored, item in zip(storage, data):
                    stored.append(item)
                obs = next_obs
                for i, transition in enumerate(transitions):
                    if transition[2]:
                        obs[i] = envs[i].reset()
                steps += args.envs
            raw, action, logp, value, reward, next_value, terminated, ended = [np.asarray(x) for x in storage]
            advantage, returns = generalized_advantage(reward, value, next_value, terminated, ended, args.gamma, args.gae_lambda)
            arrays = [raw.reshape(-1, 3, config['obs_dim']), action.reshape(-1, 3), logp.reshape(-1, 3),
                      value.ravel(), advantage.ravel(), returns.ravel()]
            batch = [torch.as_tensor(x, device=device) for x in arrays]
            stats = ppo_update(model, optimizer, batch, epochs=args.epochs, minibatch=args.minibatch,
                               clip=args.clip, entropy_coef=args.entropy, value_coef=args.value_coef)
            updates += 1
            last_batch = batch[0][:4].detach().clone()
            training_rows.append(dict(global_env_steps=steps, update=updates, **stats))
            if steps % args.checkpoint_interval == 0 or steps == args.steps:
                path = checkpoints/f'model_step{steps}.pt'
                save_checkpoint(path, model, optimizer, config, steps)
                rows = evaluate(args.task, args.curve_episodes, args.curve_seed_start, model=model, device=device)
                curve.append(dict(global_env_steps=steps, eval_episodes=len(rows), **summarize(rows)))
                csv_write(output/'learning_curve_eval.csv', curve)
                csv_write(output/'training.csv', training_rows)
                print(json.dumps(dict(task=args.task, descriptor=args.descriptor, steps=steps, **stats)), flush=True)
        final_path = checkpoints/f'model_step{steps}.pt'
        # Loading initializes random weights before replacing them: after training only.
        restored, checkpoint = load_checkpoint(final_path, device)
        with torch.no_grad():
            torch.testing.assert_close(model.distributions(last_batch).logits, restored.distributions(last_batch).logits)
            torch.testing.assert_close(model.value(last_batch), restored.value(last_batch))
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Non-finite checkpoint')
        actor_changed = [not torch.equal(old, next(a.parameters())) for old, a in zip(initial_actor, model.actors)]
        critic_changed = not torch.equal(initial_critic, next(model.critic.parameters()))
        if not all(actor_changed) or not critic_changed:
            raise RuntimeError('At least one actor/critic did not update')
        rows = evaluate(args.task, args.eval_episodes, args.eval_seed_start, model=restored, device=device)
        csv_write(output/'per_evaluation_episode_metrics.csv', rows)
        write(output/'summary.json', dict(task=args.task, method=args.descriptor, seed=args.seed,
              global_env_steps=steps, eval_episodes=len(rows), **summarize(rows)))
        if hashes != source_hashes():
            raise RuntimeError('Sources changed during run')
        write(output/'validation.json', dict(passed=True, task=args.task, descriptor=args.descriptor,
            global_env_steps=steps, updates=updates, source_hashes=hashes,
            actor_input_dim=model.actors[0].input_dim, critic_input_dim=3*config['obs_dim'],
            rollout_observation_dim=config['obs_dim'], actor_parameters_changed=actor_changed,
            critic_parameters_changed=critic_changed, checkpoint_reload_equal=True,
            wall_seconds=time.perf_counter()-started))
    except BaseException as error:
        write(output/'failure.json', dict(error=repr(error), global_env_steps=steps))
        raise
    finally:
        for env in envs:
            env.close()


def prepare(args):
    from experiments.run_adaptive_hks_supplement import prepare as prepare_core
    campaign = args.campaign.resolve()
    prepare_core(campaign)
    script = campaign/'code/experiments/run_ahks_extensions.py'
    commands = []
    for phase in ('smoke', 'formal'):
        for task in ('spread', 'sensing'):
            for descriptor in ('raw', 'fixed_hks', 'adaptive_hks'):
                for seed in ([41] if phase == 'smoke' else [41, 42, 43, 44, 45]):
                    output = campaign/'extensions'/phase/task/descriptor/f'seed_{seed}'
                    cmd = [sys.executable, str(script), 'train', '--task', task, '--descriptor', descriptor,
                           '--seed', str(seed), '--output', str(output), '--phase', phase,
                           '--steps', '2000' if phase == 'smoke' else '2000000',
                           '--eval-episodes', '3' if phase == 'smoke' else '500',
                           '--curve-episodes', '2' if phase == 'smoke' else '100',
                           '--eval-seed-start', '93000000' if phase == 'smoke' else '92000000',
                           '--curve-seed-start', '94000000' if phase == 'smoke' else '91000000',
                           '--campaign', str(campaign)]
                    if phase == 'formal':
                        cmd += ['--smoke-reference', str(campaign/'extensions/smoke'/task/descriptor/'seed_41/validation.json')]
                    commands.append(dict(phase=phase, task=task, method=descriptor, seed=seed, argv=cmd))
            for controller in (('nearest', 'hungarian') if task == 'spread' else ('nearest', 'uncertainty', 'information')):
                output = campaign/'extensions'/phase/task/('controller_'+controller)
                cmd = [sys.executable, str(script), 'controllers', '--task', task, '--controller', controller,
                       '--output', str(output), '--eval-episodes', '3' if phase == 'smoke' else '500',
                       '--eval-seed-start', '93000000' if phase == 'smoke' else '92000000', '--campaign', str(campaign)]
                commands.append(dict(phase=phase, task=task, method=controller, argv=cmd))
    write(campaign/'commands_extensions.json', commands)
    write(campaign/'commands_diagnostics.json', [dict(argv=[sys.executable,
        str(campaign/'code/experiments/diagnose_ahks_representation.py'),
        '--output', str(campaign/'extensions/representation_diagnostics')])])
    print(f'Prepared {len(commands)} extension commands, no training started.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'controllers', 'evaluate', 'prepare', 'queue', 'summarize'])
    parser.add_argument('--task', choices=['spread', 'sensing'], default='spread')
    parser.add_argument('--descriptor', choices=['raw', 'fixed_hks', 'adaptive_hks'], default='raw')
    parser.add_argument('--controller', choices=['nearest', 'hungarian', 'uncertainty', 'information'], default='nearest')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--smoke-reference', type=Path)
    parser.add_argument('--phase', choices=['smoke', 'pilot', 'formal'], default='pilot')
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--rollout', type=int, default=250)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--minibatch', type=int, default=250)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=.95)
    parser.add_argument('--gae-lambda', type=float, default=.95)
    parser.add_argument('--clip', type=float, default=.2)
    parser.add_argument('--entropy', type=float, default=.01)
    parser.add_argument('--value-coef', type=float, default=.5)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--checkpoint-interval', type=int, default=20000)
    parser.add_argument('--eval-episodes', type=int, default=3)
    parser.add_argument('--curve-episodes', type=int, default=2)
    parser.add_argument('--eval-seed-start', type=int, default=93000000)
    parser.add_argument('--curve-seed-start', type=int, default=94000000)
    args = parser.parse_args()
    if args.campaign is not None and args.mode not in ('prepare', 'queue', 'summarize'):
        from experiments.run_adaptive_hks_supplement import verify
        campaign = args.campaign.resolve()
        if ROOT != campaign/'code':
            raise RuntimeError('Campaign jobs must run from the frozen code directory')
        verify(campaign, json.loads((campaign/'manifest.json').read_text()))
    if args.mode == 'prepare':
        if args.campaign is None:
            parser.error('--campaign required')
        prepare(args)
    elif args.mode == 'queue':
        if args.campaign is None or args.phase == 'pilot':
            parser.error('queue requires --campaign and --phase smoke/formal')
        for item in json.loads((args.campaign/'commands_extensions.json').read_text()):
            if item['phase'] == args.phase:
                subprocess.run(item['argv'], check=True, cwd=args.campaign)
    elif args.mode == 'summarize':
        if args.campaign is None:
            parser.error('--campaign required')
        summarize_campaign(args.campaign)
    else:
        if args.output is None:
            parser.error('--output required')
        if args.mode == 'train':
            train(args)
        else:
            if args.eval_episodes <= 0:
                parser.error('--eval-episodes must be positive')
            torch.set_num_threads(args.threads)
            model = None
            if args.mode == 'evaluate':
                if args.checkpoint is None:
                    parser.error('--checkpoint required')
                model, checkpoint = load_checkpoint(args.checkpoint)
                if checkpoint['config']['task'] != args.task:
                    raise ValueError('Checkpoint task mismatch')
            elif args.controller not in (('nearest', 'hungarian') if args.task == 'spread' else ('nearest', 'uncertainty', 'information')):
                parser.error('Controller not supported for this task')
            args.output.mkdir(parents=True, exist_ok=False)
            rows = evaluate(args.task, args.eval_episodes, args.eval_seed_start, model=model, controller=args.controller)
            csv_write(args.output/'per_evaluation_episode_metrics.csv', rows)
            write(args.output/'summary.json', dict(task=args.task, method=args.controller if model is None else checkpoint['config']['descriptor'],
                  eval_episodes=len(rows), **summarize(rows)))
            write(args.output/'protocol.json', dict(task=args.task, controller=args.controller if model is None else None,
                  checkpoint=str(args.checkpoint), seed_start=args.eval_seed_start, source_hashes=source_hashes(),
                  sensing_config=asdict(SensingConfig()) if args.task == 'sensing' else None))
            print(json.dumps(summarize(rows)), flush=True)


def summarize_campaign(campaign):
    from experiments.summarize_adaptive_hks_supplement import estimate
    output = {}
    for task in ('spread', 'sensing'):
        task_result = {'methods': {}, 'paired_differences': {}, 'missing': [], 'controllers': {}}
        all_rows = {}
        for method in ('raw', 'fixed_hks', 'adaptive_hks'):
            rows = {}
            for seed in (41, 42, 43, 44, 45):
                folder = campaign/'extensions/formal'/task/method/f'seed_{seed}'
                if not (folder/'validation.json').exists():
                    task_result['missing'].append([method, seed])
                    continue
                validation = json.loads((folder/'validation.json').read_text())
                row = json.loads((folder/'summary.json').read_text())
                if not validation['passed'] or row['global_env_steps'] != 2000000 or row['eval_episodes'] != 500:
                    raise ValueError('Invalid formal result: '+str(folder))
                rows[seed] = row
            all_rows[method] = rows
            if rows:
                keys = [k for k in next(iter(rows.values())) if k not in ('task', 'method', 'seed', 'global_env_steps', 'eval_episodes')]
                task_result['methods'][method] = {k: estimate([r[k] for r in rows.values()]) for k in keys}
        for control in ('raw', 'fixed_hks'):
            common = sorted(all_rows['adaptive_hks'].keys() & all_rows[control].keys())
            if common:
                task_result['paired_differences']['adaptive_hks_minus_'+control] = {
                    k: estimate([all_rows['adaptive_hks'][s][k]-all_rows[control][s][k] for s in common])
                    for k in task_result['methods'][control]}
        for controller in (('nearest', 'hungarian') if task == 'spread' else ('nearest', 'uncertainty', 'information')):
            path = campaign/'extensions/formal'/task/('controller_'+controller)/'summary.json'
            if path.exists():
                task_result['controllers'][controller] = json.loads(path.read_text())
        task_result['complete_training_matrix'] = not task_result['missing']
        task_result['complete_controllers'] = len(task_result['controllers']) == (2 if task == 'spread' else 3)
        output[task] = task_result
    write(campaign/'aggregate_extensions.json', output)


if __name__ == '__main__':
    main()
