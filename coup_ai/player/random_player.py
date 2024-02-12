from jaxtyping import PRNGKeyArray, Scalar, Bool
from ..gamerules.step import Action


def choice_action(model_state, rng_key: PRNGKeyArray, obs) -> Action:
    pass


def choice_challenge(model_state, rng_key: PRNGKeyArray, obs) -> Bool[Scalar, ""]:
    pass


def choice_block(model_state, rng_key: PRNGKeyArray, obs) -> Bool[Scalar, ""]:
    pass


def choice_exchange(model_state, rng_key: PRNGKeyArray, obs):
    pass
