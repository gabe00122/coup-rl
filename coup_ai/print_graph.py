import jax
from jax import random, numpy as jnp
from jaxlib import xla_client
from .gamerules.game import initial_state
from .gamerules.step import action_phase, challenge_phase
from .gamerules.deck import draw_card
from .gamerules.actions import preform_action


def main():
    # comp = jax.xla_computation(initial_state, static_argnums=(0,))(6, random.PRNGKey(123))
    # dot_graph = comp.as_hlo_dot_graph()
    rng_key = random.PRNGKey(345)
    state = initial_state(jnp.uint8(6), rng_key)
    # state = action_phase(state, )

    compiled_text = jax.jit(initial_state).lower(
        jnp.uint8(6), rng_key
    ).compile().as_text()

    dot_graph = xla_client._xla.hlo_module_to_dot_graph(xla_client._xla.hlo_module_from_text(compiled_text))
    with open("out.dot", 'w') as f:
        f.write(dot_graph)


if __name__ == '__main__':
    main()
