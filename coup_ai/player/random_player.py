from jax import random, numpy as jnp
from jaxtyping import PRNGKeyArray, Scalar, Array, Bool, UInt8
from typing import TypedDict
from ..gamerules.step import Action


# zero index is the active player
class ActionObs(TypedDict):
    active_players: Bool[Array, "6"]
    coins: UInt8[Scalar, ""]


def choice_action(model_state, rng_key: PRNGKeyArray, obs: ActionObs) -> Action:
    player_coins = obs["coins"][0]

    random.choice()
    pass


def choice_challenge(model_state, rng_key: PRNGKeyArray, obs) -> Bool[Scalar, ""]:
    pass


def choice_block(model_state, rng_key: PRNGKeyArray, obs) -> Bool[Scalar, ""]:
    pass


def choice_exchange(model_state, rng_key: PRNGKeyArray, obs):
    pass


def choice_bool(rng_key: PRNGKeyArray, bools: Bool[Array, "n"]):
    count = jnp.count_nonzero(bools)
    probs = bools / count
