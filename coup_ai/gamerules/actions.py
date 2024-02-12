import jax
from jax import numpy as jnp
from jaxtyping import UInt8, Array, Scalar
from .game import GameState
from .step import eliminate_card


income = jnp.uint8(0)
foreign_aid = jnp.uint8(1)
coup = jnp.uint8(2)
tax = jnp.uint8(3)
exchange = jnp.uint8(4)
steal = jnp.uint8(5)
assassinate = jnp.uint8(6)


def preform_action(state: GameState) -> GameState:
    action_kind = state["action"]
    return jax.lax.switch(action_kind, [
        income_fn,
        foreign_aid_fn,
        coup_fn,
        tax_fn,
        exchange_fn,
        steal_fn,
        assassinate_fn,
    ], state)


def income_fn(state: GameState) -> GameState:
    return change_coins(state, state["active_player"], jnp.uint8(1))


def foreign_aid_fn(state: GameState) -> GameState:
    return change_coins(state, state["active_player"], jnp.uint8(2))


def coup_fn(state: GameState) -> GameState:
    state = change_coins(state, state["active_player"], jnp.uint8(-7))
    state = eliminate_card(state, state["target"])

    return state


def tax_fn(state: GameState) -> GameState:
    return change_coins(state, state["active_player"], jnp.uint8(3))


def exchange_fn(state: GameState) -> GameState:
    return state


def steal_fn(state: GameState) -> GameState:
    active_index = state["active_player"]
    target_index = state["target"]
    coins = state["players"]["coins"]

    amount = jnp.minimum(coins[target_index], jnp.uint8(2))
    state = change_coins(state, active_index, amount)
    state = change_coins(state, target_index, -amount)

    return state


def assassinate_fn(state: GameState) -> GameState:
    active_index = state["active_player"]
    state = change_coins(state, active_index, jnp.uint8(-3))
    state = eliminate_card(state, state["target"])

    return state


def change_coins(state: GameState, player_index: UInt8[Scalar, ""], amount: UInt8[Scalar, ""]) -> GameState:
    players = state["players"]
    coins = players["coins"]

    coins = coins.at[player_index].set(coins[player_index] + amount)

    players |= {"coins": coins}
    return state | {"players": players}
