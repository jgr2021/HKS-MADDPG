import itertools
import numpy as np

from utils.make_env import make_env


N_EPISODES = 20
EPISODE_LENGTH = 25
SEED = 2040
COVERAGE_THRESHOLD = 0.1


def get_metrics(env):
    agents = env.world.agents
    landmarks = env.world.landmarks

    distances = np.array([
        [
            np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
            for landmark in landmarks
        ]
        for agent in agents
    ])

    min_dist_per_landmark = distances.min(axis=0)
    occupied = int(np.sum(min_dist_per_landmark < COVERAGE_THRESHOLD))
    min_dist_sum = float(min_dist_per_landmark.sum())

    return occupied, min_dist_sum


def best_initial_assignment(env):
    """
    在每个 episode 开始时，枚举 3! = 6 种分配，
    找到 agent 到 landmark 总距离最小的一对一匹配。
    """
    agent_positions = np.array([
        agent.state.p_pos for agent in env.world.agents
    ])
    landmark_positions = np.array([
        landmark.state.p_pos for landmark in env.world.landmarks
    ])

    best_perm = None
    best_cost = float("inf")

    for perm in itertools.permutations(range(3)):
        cost = sum(
            np.linalg.norm(agent_positions[i] - landmark_positions[perm[i]])
            for i in range(3)
        )
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    return best_perm


def cardinal_action(agent, target_position):
    """
    简单 PD 控制器：
    - 朝 assigned landmark 移动；
    - 接近目标时利用速度项主动刹车；
    - 最终输出 MPE 所需的五维 one-hot 动作。

    动作编码：
    [no-op, +x, -x, +y, -y]
    """
    error = target_position - agent.state.p_pos
    velocity = agent.state.p_vel

    # 位置项推动靠近目标；速度项用于减小过冲
    control = 5.5 * error - 2.8 * velocity

    # 已足够接近且速度足够低，则不动
    if np.linalg.norm(error) < 0.045 and np.linalg.norm(velocity) < 0.08:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # 五动作只能沿单一坐标轴移动，选绝对控制量更大的坐标轴
    axis = int(np.argmax(np.abs(control)))

    action = np.zeros(5, dtype=np.float32)

    if axis == 0:
        action[1 if control[0] >= 0 else 2] = 1.0
    else:
        action[3 if control[1] >= 0 else 4] = 1.0

    return action


def main():
    # 必须和你训练的离散 MADDPG 版本一致
    env = make_env("simple_spread", discrete_action=True)
    env.seed(SEED)
    np.random.seed(SEED)

    results = []

    for ep_i in range(N_EPISODES):
        env.reset()

        assignment = best_initial_assignment(env)
        max_coverage = 0

        for _ in range(EPISODE_LENGTH):
            actions = []

            for agent_i, agent in enumerate(env.world.agents):
                target = env.world.landmarks[assignment[agent_i]].state.p_pos
                actions.append(cardinal_action(agent, target))

            env.step(actions)

            occupied, _ = get_metrics(env)
            max_coverage = max(max_coverage, occupied)

        final_coverage, final_dist_sum = get_metrics(env)

        results.append({
            "max_coverage": max_coverage,
            "final_coverage": final_coverage,
            "final_dist_sum": final_dist_sum,
        })

        print(
            f"[Greedy {ep_i + 1:02d}] "
            f"max_coverage={max_coverage}/3 | "
            f"final_coverage={final_coverage}/3 | "
            f"final_min_dist_sum={final_dist_sum:.3f}"
        )

    mean_max = np.mean([x["max_coverage"] for x in results])
    mean_final = np.mean([x["final_coverage"] for x in results])
    mean_dist = np.mean([x["final_dist_sum"] for x in results])

    max3 = sum(x["max_coverage"] == 3 for x in results)
    final3 = sum(x["final_coverage"] == 3 for x in results)

    print("\n" + "=" * 72)
    print("Greedy reachability summary")
    print("=" * 72)
    print(f"mean max coverage:       {mean_max:.2f}/3")
    print(f"mean final coverage:     {mean_final:.2f}/3")
    print(f"max3 count:              {max3}/{N_EPISODES}")
    print(f"final3 count:            {final3}/{N_EPISODES}")
    print(f"mean final distance sum: {mean_dist:.3f}")

    env.close()


if __name__ == "__main__":
    main()