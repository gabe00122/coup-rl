import jax
from jax import random, numpy as jnp
from jax.experimental import checkify
from jaxtyping import Bool, Scalar
from .gamerules.game import initial_state, GameState
from .display.print import print_players_state, print_deck
from .gamerules.step import action_phase, challenge_phase, lose_card_phase, eliminate_card
from .gamerules.actions import tax
from .gamerules.actions import preform_action


def is_challengeable(state: GameState) -> Bool[Scalar, ""]:
    return state["action"] >= 3


@jax.jit
def test() -> GameState:
    rng_key = random.PRNGKey(0)
    state: GameState = initial_state(jnp.uint8(4), rng_key)

    state = action_phase(state, {"kind": tax, "target": jnp.uint8(0)})
    state = jax.lax.cond(
        is_challengeable(state),
        lambda: challenge_phase(state, jnp.array([False, False, False, False, False, False]), rng_key), # todo split rng_key
        lambda: state
    )

    state = jax.lax.cond(
        jnp.logical_and(state["action_alive"], jnp.logical_not(state["block_alive"])),
        lambda: preform_action(state),
        lambda: state,
    )

    # todo block when blockable

    def on_discard_choice_queued():
        return state

    state = jax.lax.cond(
        state["discard_choice_queued"],
        on_discard_choice_queued,
        lambda: state,
    )

    return state


def main():
    err, state = checkify.checkify(test)()
    err.throw()

    print_players_state(state["players"])
    print_deck(state["deck"])
    print(state)


if __name__ == '__main__':
    main()
