"""Bounded allocation/sampling diagnostic, not a training run."""
import ctypes
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'experiments/gsp_long_20260906b/code'


def memory():
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ('total_phys', 'avail_phys', 'total_page', 'avail_page',
             'total_virtual', 'avail_virtual', 'extra')]
    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return {name: getattr(status, name) for name in
            ('avail_phys', 'total_page', 'avail_page', 'load')}


def worker(index, events, release):
    try:
        sys.path.insert(0, str(FROZEN))
        import numpy as np
        import torch
        from utils.buffer import ReplayBuffer
        torch.set_num_threads(6)
        np.random.seed(index)
        replay = ReplayBuffer(1000000, 3, [18] * 3, [5] * 3)
        arrays = (replay.obs_buffs + replay.ac_buffs + replay.rew_buffs
                  + replay.next_obs_buffs + replay.done_buffs)
        for array in arrays:
            array.fill(1.0)
        for array in replay.rew_buffs:
            array[::2] = 1.5
        replay.filled_i = replay.max_steps
        torch.zeros(1, device='cuda')
        events.put({'event': 'ready', 'worker': index, 'pid': os.getpid(),
                    'buffer_bytes': sum(a.nbytes for a in arrays)})
        if not release.wait(90):
            raise TimeoutError('Peer readiness timeout')
        start = time.perf_counter()
        for _ in range(24):
            sample = replay.sample(1024, to_gpu=True, norm_rews=True)
            assert all(torch.isfinite(t).all().item() for group in sample for t in group)
        torch.cuda.synchronize()
        events.put({'event': 'passed', 'worker': index,
                    'sample_calls': 24, 'seconds': time.perf_counter() - start,
                    'tensor_shapes': [[list(t.shape) for t in group] for group in sample],
                    'device': str(sample[0][0].device)})
    except Exception:
        events.put({'event': 'failed', 'worker': index, 'error': traceback.format_exc()})
        raise


def main():
    before = memory()
    if before['avail_phys'] < 6 * 1024**3 or before['avail_page'] < 10 * 1024**3:
        raise RuntimeError('Insufficient headroom to start diagnostic')
    ctx = mp.get_context('spawn')
    events, release = ctx.Queue(), ctx.Event()
    workers = [ctx.Process(target=worker, args=(i, events, release)) for i in (1, 2)]
    records, readings = [], [before]
    try:
        for process in workers:
            process.start()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            readings.append(memory())
            try:
                record = events.get(timeout=0.2)
                records.append(record)
                print(json.dumps(record), flush=True)
                if record['event'] == 'failed':
                    break
                if sum(r['event'] == 'ready' for r in records) == 2:
                    release.set()
                if sum(r['event'] == 'passed' for r in records) == 2:
                    break
            except queue.Empty:
                if all(not p.is_alive() for p in workers):
                    break
        for process in workers:
            process.join(timeout=10)
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
    result = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'passed': sum(r['event'] == 'passed' for r in records) == 2
                  and all(p.exitcode == 0 for p in workers),
        'scope': 'Two full frozen replay buffers and CUDA sampling only; no model updates or environment steps',
        'limitations': 'Short synthetic diagnostic does not prove long-run peak memory safety or uniquely explain the prior failure',
        'training_steps': 0, 'frozen_source': str(FROZEN),
        'before': before, 'after': memory(),
        'minimum_available_physical_bytes': min(m['avail_phys'] for m in readings),
        'minimum_available_commit_bytes': min(m['avail_page'] for m in readings),
        'events': records, 'exit_codes': [p.exitcode for p in workers]}
    output = ROOT / 'experiments/system_memory_20260907' / ('post_reboot_validation_' + time.strftime('%H%M%S') + '.json')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'artifact': str(output), **result}), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
