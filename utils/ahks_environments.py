"""Legacy-MPE adapter and a belief-only cooperative active sensing task.

Actions in both tasks are [stay, +x, -x, +y, -y]. Active sensing uses
an explicitly shared team Kalman belief, not communication-free sensing.
"""
import itertools
from dataclasses import dataclass, asdict
import numpy as np

CONTROLS = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)


def assignment(cost):
    """Exact rectangular assignment for three agents; same optimum as Hungarian."""
    cost = np.asarray(cost)
    if cost.shape[0] != 3 or cost.shape[1] < 3:
        raise ValueError('Expected three agents and at least three targets')
    return np.array(min(itertools.permutations(range(cost.shape[1]), 3),
                        key=lambda p: sum(cost[i, p[i]] for i in range(3))))


def pd_actions(error, velocity):
    control = 5.5 * error - 2.8 * velocity
    axis = np.abs(control).argmax(-1)
    selected = control[np.arange(3), axis]
    actions = 1 + 2 * axis + (selected < 0)
    actions[(np.linalg.norm(error, axis=-1) < .045) &
            (np.linalg.norm(velocity, axis=-1) < .08)] = 0
    return actions.astype(np.int64)


def navigation_controller(obs, kind):
    # All geometry is obtained from the allowed observation, never simulator state.
    agents = obs[:, 2:4]
    targets = obs[0, 4:10].reshape(3, 2) + agents[0]
    distances = np.linalg.norm(agents[:, None] - targets[None], axis=-1)
    selected = distances.argmin(-1) if kind == 'nearest' else assignment(distances)
    return pd_actions(targets[selected] - agents, obs[:, :2])


class SpreadEnv:
    obs_dim = 18
    horizon = 25
    task = 'spread'

    def __init__(self, seed=0):
        from utils.make_env import make_env
        state = np.random.get_state()
        try:
            self.env = make_env('simple_spread', discrete_action=True)
        finally:
            np.random.set_state(state)
        self.rng = np.random.RandomState(seed)

    def _local_call(self, fn, *args):
        state = np.random.get_state()
        np.random.set_state(self.rng.get_state())
        try:
            return fn(*args)
        finally:
            self.rng.set_state(np.random.get_state())
            np.random.set_state(state)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.elapsed = 0
        self.collision_steps = self.max_coverage = 0
        self.min_separation = float('inf')
        self.total_return = 0.
        return np.asarray(self._local_call(self.env.reset), dtype=np.float32)

    def step(self, actions):
        actions = check_actions(actions)
        obs, rewards, dones, _ = self._local_call(self.env.step, np.eye(5)[actions].copy())
        self.elapsed += 1
        reward = float(np.mean(rewards))
        self.total_return += reward
        agents = np.array([a.state.p_pos for a in self.env.world.agents])
        landmarks = np.array([a.state.p_pos for a in self.env.world.landmarks])
        d = np.linalg.norm(agents[:, None] - landmarks[None], axis=-1)
        separations = [np.linalg.norm(agents[i]-agents[j]) for i, j in itertools.combinations(range(3), 2)]
        collisions = sum(sep < self.env.world.agents[i].size + self.env.world.agents[j].size
                         for sep, (i, j) in zip(separations, itertools.combinations(range(3), 2)))
        nearest = d.min(0)
        coverage = int((nearest < .10).sum())
        self.collision_steps += int(collisions > 0)
        self.max_coverage = max(self.max_coverage, coverage)
        self.min_separation = min(self.min_separation, min(separations))
        radii = np.linspace(.05, .30, 26)
        metrics = dict(
            **{'return': self.total_return},
            hungarian_assignment_distance=float(d[np.arange(3), assignment(d)].mean()),
            coverage_radius_auc=float(np.trapz([(nearest < r).mean() for r in radii], radii)/.25),
            collision_step_rate=self.collision_steps/self.elapsed,
            final_coverage=coverage, max_coverage=self.max_coverage,
            final3=int(coverage == 3), max3=int(self.max_coverage == 3),
            collisions=int(collisions), nearest_landmark_distance=float(nearest.mean()),
            minimum_agent_separation=float(self.min_separation),
            unique_landmarks_covered=int(len(set(d.argmin(1)))), full_coverage_rate=int(coverage == 3))
        terminated = bool(all(dones))
        done = terminated or self.elapsed >= self.horizon
        return np.asarray(obs, dtype=np.float32), reward, done, {
            'terminated': terminated, 'truncated': done and not terminated, 'metrics': metrics}

    def close(self):
        self.env.close()


def check_actions(actions):
    actions = np.asarray(actions)
    if actions.shape != (3,) or not np.issubdtype(actions.dtype, np.integer) or np.any((actions < 0) | (actions > 4)):
        raise ValueError('Expected three integer action indices in [0,4]')
    return actions


@dataclass(frozen=True)
class SensingConfig:
    horizon: int = 100
    dt: float = .1
    sensing_range: float = .7
    measurement_std: float = .1
    acceleration_std: float = .08
    move_weight: float = .05
    collision_weight: float = .2
    collision_distance: float = .30
    loss_after_steps: int = 10

    def __post_init__(self):
        if min(self.horizon, self.dt, self.sensing_range, self.measurement_std,
               self.acceleration_std, self.collision_distance, self.loss_after_steps) <= 0:
            raise ValueError('Task sizes and noise scales must be positive')


def kalman_update(mean, covariance, measurement, noise):
    """Position-only observation; Joseph covariance update for numerical stability."""
    h = np.eye(4)[:2]
    innovation = h @ covariance @ h.T + noise
    gain = np.linalg.solve(innovation, h @ covariance).T
    updated = mean + gain @ (measurement - h @ mean)
    residual = np.eye(4) - gain @ h
    posterior = residual @ covariance @ residual.T + gain @ noise @ gain.T
    return updated, .5 * (posterior + posterior.T)


class ActiveSensingEnv:
    task = 'sensing'
    # 18 geometry + 3*(velocity2 + covariance16 + age1 + seen1) + remaining-time1.
    obs_dim = 79

    def __init__(self, seed=0, config=None):
        self.config = config or SensingConfig()
        self.horizon = self.config.horizon
        self.rng = np.random.RandomState(seed)
        dt = self.config.dt
        self.F = np.eye(4)
        self.F[:2, 2:] = dt * np.eye(2)
        self.G = np.vstack((.5 * dt**2 * np.eye(2), dt * np.eye(2)))
        self.Q = self.config.acceleration_std**2 * self.G @ self.G.T
        self.R = self.config.measurement_std**2 * np.eye(2)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.elapsed = 0
        self.positions = self.rng.uniform(-1, 1, (3, 2))
        self.velocities = np.zeros((3, 2))
        # Sample truth from a public prior: the belief is not initialized with truth.
        self.mean = np.column_stack((self.rng.uniform(-1, 1, (3, 2)), np.zeros((3, 2))))
        self.covariance = np.tile(np.diag([.20, .20, .02, .02]), (3, 1, 1))
        self.truth = np.array([self.rng.multivariate_normal(m, p) for m, p in zip(self.mean, self.covariance)])
        # Exogenous noise indexed by time/agent/target; policy-dependent visibility
        # cannot change the future noise stream in paired evaluation.
        self.process_noise = self.rng.normal(0, self.config.acceleration_std, (self.horizon, 3, 2))
        self.measurement_noise = self.rng.normal(0, self.config.measurement_std, (self.horizon, 3, 3, 2))
        self.age = np.zeros(3, dtype=int)
        self.seen = np.zeros(3, dtype=bool)
        self.loss_streak = np.zeros(3, dtype=int)
        self.max_loss_streak = 0
        self.totals = dict(reward=0., squared_error=0., uncertainty=0., collision=0.,
                           travel=0., loss=0., nees=0.)
        return self.observation()

    def observation(self):
        extra = np.concatenate((self.mean[:, 2:], self.covariance.reshape(3, 16),
                                self.age[:, None]/self.horizon, self.seen[:, None]), axis=1).ravel()
        obs = []
        for i in range(3):
            other = [j for j in range(3) if j != i]
            obs.append(np.concatenate((self.velocities[i], self.positions[i],
                (self.mean[:, :2]-self.positions[i]).ravel(),
                (self.positions[other]-self.positions[i]).ravel(), np.zeros(4),
                extra, [1-self.elapsed/self.horizon])))
        return np.asarray(obs, dtype=np.float32)

    def step(self, actions):
        actions = check_actions(actions)
        if self.elapsed >= self.horizon:
            raise RuntimeError('Reset after episode termination')
        previous = self.positions.copy()
        self.velocities = .75*self.velocities + .5*CONTROLS[actions]
        self.positions += self.config.dt*self.velocities
        self.truth = self.truth @ self.F.T + self.process_noise[self.elapsed] @ self.G.T
        self.mean = self.mean @ self.F.T
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        visible = np.linalg.norm(self.positions[:, None]-self.truth[None, :, :2], axis=-1) <= self.config.sensing_range
        self.seen = visible.any(0)
        self.age = np.where(self.seen, 0, self.age+1)
        for i, j in zip(*np.where(visible)):
            measurement = self.truth[j, :2] + self.measurement_noise[self.elapsed, i, j]
            self.mean[j], self.covariance[j] = kalman_update(self.mean[j], self.covariance[j], measurement, self.R)
        uncertainty = float(np.trace(self.covariance, axis1=1, axis2=2).mean())
        travel = float(np.linalg.norm(self.positions-previous, axis=-1).mean())
        collision = any(np.linalg.norm(self.positions[i]-self.positions[j]) < self.config.collision_distance
                        for i, j in itertools.combinations(range(3), 2))
        reward = -uncertainty - self.config.move_weight*travel - self.config.collision_weight*collision
        error = self.truth-self.mean
        loss = self.age >= self.config.loss_after_steps
        self.loss_streak = np.where(loss, self.loss_streak+1, 0)
        self.max_loss_streak = max(self.max_loss_streak, int(self.loss_streak.max()))
        for key, value in dict(reward=reward, squared_error=float((error[:, :2]**2).sum(-1).mean()),
                               uncertainty=uncertainty, collision=float(collision), travel=travel,
                               loss=float(loss.mean()),
                               nees=float(np.mean([e @ np.linalg.solve(p, e) for e, p in zip(error, self.covariance)]))).items():
            self.totals[key] += value
        self.elapsed += 1
        metrics = dict(**{'return': self.totals['reward']},
            tracking_rmse=float(np.sqrt(self.totals['squared_error']/self.elapsed)),
            posterior_uncertainty=self.totals['uncertainty']/self.elapsed,
            target_loss_fraction=self.totals['loss']/self.elapsed,
            max_target_loss_duration=self.max_loss_streak*self.config.dt,
            collision_step_rate=self.totals['collision']/self.elapsed,
            travel_cost=self.totals['travel'], nees=self.totals['nees']/self.elapsed)
        done = self.elapsed == self.horizon
        return self.observation(), reward, done, dict(terminated=done, truncated=False, metrics=metrics)

    def close(self):
        pass


def sensing_controller(obs, kind, config=None):
    config = config or SensingConfig()
    agents, velocity = obs[:, 2:4], obs[:, :2]
    targets = agents[0] + obs[0, 4:10].reshape(3, 2)
    belief = obs[0, 18:78].reshape(3, 20)
    covariance = belief[:, 2:18].reshape(3, 4, 4).astype(float)
    if kind == 'nearest':
        selected = np.linalg.norm(agents[:, None]-targets[None], axis=-1).argmin(-1)
        return pd_actions(targets[selected]-agents, velocity)
    if kind == 'uncertainty':
        selected = int(np.trace(covariance, axis1=1, axis2=2).argmax())
        return pd_actions(targets[selected]-agents, velocity)
    if kind != 'information':
        raise ValueError(kind)
    f = np.eye(4)
    f[:2, 2:] = config.dt*np.eye(2)
    g = np.vstack((.5*config.dt**2*np.eye(2), config.dt*np.eye(2)))
    prediction = f @ covariance @ f.T + config.acceleration_std**2*g @ g.T
    target_next = targets + config.dt*belief[:, :2]
    candidate = agents[:, None] + config.dt*(.75*velocity[:, None]+.5*CONTROLS[None])
    distance = np.linalg.norm(candidate[:, :, None]-target_next[None, None], axis=-1)
    reduction = np.array([np.trace(p)-np.trace(kalman_update(np.zeros(4), p, np.zeros(2),
                                     config.measurement_std**2*np.eye(2))[1]) for p in prediction])
    # One sensor per assigned target. Predicted-mean range approximation, explicitly
    # not an exact visibility-probability integral or globally optimal joint sensing.
    score = (distance <= config.sensing_range)*reduction[None, None] - 1e-6*distance
    best_actions = score.argmax(1)
    best_scores = score.max(1)
    selected = assignment(-best_scores)
    return best_actions[np.arange(3), selected].astype(np.int64)


def make_task(task, seed=0, sensing_config=None):
    if task == 'spread':
        return SpreadEnv(seed)
    if task == 'sensing':
        return ActiveSensingEnv(seed, sensing_config)
    raise ValueError(task)
