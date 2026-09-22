"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.

GSP-MADDPG v1 adds exactly one new environment alias:
    simple_spread_gsp_v1

It creates the original MPE ``simple_spread`` world and then wraps only its
observations from 18D to 26D. The original ``simple_spread`` path remains
unchanged and is still available for Vanilla MADDPG.
"""

import re


GSP_V1_ENV_ID = "simple_spread_gsp_v1"
GSP_V1_BASE_SCENARIO = "simple_spread"
SIMPLE_SPREAD_6X6_ENV_ID = "simple_spread_6x6"
SIMPLE_SPREAD_NXN_PATTERN = re.compile(r"^simple_spread_(\d+)x\1$")
GSP_ENV_ALIASES = {
    GSP_V1_ENV_ID: "full",
    "simple_spread_gsp_v2a_spectrum": "spectrum",
    "simple_spread_gsp_v2a_hks": "hks",
    "simple_spread_gsp_v2a_edges": "edges",
    "simple_spread_gsp_v2a_full_edges": "full_edges",
    "simple_spread_gsp_v2d_spectral_role": "spectral_role",
    "simple_spread_gsp_v2e_all_node_roles": "all_node_roles",
    "simple_spread_gsp_topology_3node_aa_hks": "topology_3node_aa_hks",
    "simple_spread_gsp_topology_6node_al_hks": "topology_6node_al_hks",
    "simple_spread_gsp_topology_6node_aal_hks": "topology_6node_aal_hks",
}


def make_env(scenario_name, benchmark=False, discrete_action=False):
    """Create a MPE environment.

    Parameters
    ----------
    scenario_name : str
        Any original MPE scenario name, or ``simple_spread_gsp_v1``.
    benchmark : bool
        Preserved original MPE behavior.
    discrete_action : bool
        Preserved original action-space choice.
    """
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    gsp_mode = GSP_ENV_ALIASES.get(scenario_name)
    use_gsp_wrapper = gsp_mode is not None
    cardinality_match = SIMPLE_SPREAD_NXN_PATTERN.fullmatch(scenario_name)
    cardinality = int(cardinality_match.group(1)) if cardinality_match else None
    if cardinality is not None and cardinality < 2:
        raise ValueError("simple_spread NxN cardinality must be at least two")
    use_nxn = cardinality is not None
    source_scenario = (
        GSP_V1_BASE_SCENARIO if (use_gsp_wrapper or use_nxn) else scenario_name
    )

    # Load the original scenario; GSP-v1 never changes reward or physics.
    scenario = scenarios.load(source_scenario + ".py").Scenario()
    world = scenario.make_world()
    if use_nxn:
        # Separate cardinality configurations of the original task. Physics,
        # reset, observation, reward, collision sizes, and actions are exactly
        # the original simple_spread callbacks/properties.
        from multiagent.core import Agent, Landmark
        world.agents = [Agent() for _ in range(cardinality)]
        for index, agent in enumerate(world.agents):
            agent.name = "agent %d" % index
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        world.landmarks = [Landmark() for _ in range(cardinality)]
        for index, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % index
            landmark.collide = False
            landmark.movable = False
        scenario.reset_world(world)

    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
            discrete_action=discrete_action,
        )
    else:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            discrete_action=discrete_action,
        )

    if use_gsp_wrapper:
        from utils.gsp_env_wrapper import GSPObservationWrapper
        env = GSPObservationWrapper(env, mode=gsp_mode)

    return env
