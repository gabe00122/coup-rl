from .game import GameState
import jax
from jax import numpy as jnp
from jax.experimental import checkify
from typing import TypedDict
from jaxtyping import UInt8, Scalar, Array, Bool, PRNGKeyArray
from .actions import coup, assassinate
from .cards import duke, ambassador, captain, assassin
from .deck import replace_card


class Action(TypedDict):
    kind: UInt8[Scalar, ""]
    target: UInt8[Scalar, ""]


def step(state: GameState, action: Action) -> GameState:
    return jax.lax.switch(state["phase"], [
        action_phase
    ], state, action)


def action_phase(state: GameState, action: Action) -> GameState:
    active_player_index = state["active_player"]
    coins = state["players"]["coins"][active_player_index]

    checkify.check(
        jnp.logical_not(jnp.logical_and(action["kind"] == coup, coins < 7)),
        "Coup requires 7 coins"
    )
    checkify.check(
        jnp.logical_not(jnp.logical_and(action["kind"] == assassinate, coins < 3)),
        "Assassinate requires 3 coins"
    )

    return state | {
        "phase": jnp.uint8(1),
        "action": action["kind"],
        "target": action["target"],
    }


def challenge_phase(state: GameState, challenges: Bool[Array, "players"], rng_key: PRNGKeyArray) -> GameState:
    # find the first challenge checking clockwise from the active player
    active_player_index = state["active_player"]
    player_count = len(state["players"])

    def fori_fn(i: int, carry: tuple[int, bool]):
        _, found = carry
        index = (active_player_index + i) % player_count
        return jax.lax.cond(
            jnp.logical_and(jnp.logical_not(found), challenges[index]),
            lambda: (jnp.uint8(index), True),
            lambda: carry,
        )

    challenger_index, any_challenger = jax.lax.fori_loop(
        1,
        player_count + 1,
        fori_fn,
        (jnp.uint8(0), False)
    )

    def some_challenge():
        active_player_cards = state["players"]["cards"][active_player_index]

        needed_card = jax.lax.select_n(
            state["action"],
            jnp.full((2,), jnp.uint8(0)),  # income - not possible to challenge
            jnp.full((2,), jnp.uint8(0)),  # foreign_aid - not possible to challenge
            jnp.full((2,), jnp.uint8(0)),  # coup - not possible to challenge
            jnp.full((2,), duke),  # tax
            jnp.full((2,), ambassador),  # exchange
            jnp.full((2,), captain),  # steal
            jnp.full((2,), assassin),  # assassinate
        )
        is_truth = active_player_cards == needed_card

        def on_bluff():
            return eliminate_card(state, state["active_player"])

        def on_truth():
            replace_idx = jnp.argmax(is_truth)
            card = active_player_cards[replace_idx]
            new_deck, card = replace_card(state["deck"], rng_key, card)

            new_cards = state["players"]["cards"].at[active_player_index, replace_idx].set(card)
            players = state["players"] | {"cards": new_cards}

            return state | {"players": players, "deck": new_deck}

        return jax.lax.cond(is_truth.any(), on_truth, on_bluff)

    def no_challenge():
        return state

    return jax.lax.cond(any_challenger, some_challenge, no_challenge)


def eliminate_card(state: GameState, player_index: UInt8[Scalar, ""]) -> GameState:
    cards_faceup = state["players"]["cards_faceup"]

    def none_faceup():
        return state | {
            "discard_choice_target": player_index,
            "discard_choice_queued": jnp.bool(True)
        }

    def any_faceup():
        new_cards_faceup = cards_faceup.at[player_index].set(True)
        players = state["players"] | {"cards_faceup": new_cards_faceup}
        return state | {"players": players}

    return jax.lax.cond(cards_faceup[player_index].any(), any_faceup, none_faceup)


def lose_card_phase(state: GameState, choice: UInt8[Scalar, ""]) -> GameState:
    target = state["discard_choice_target"]
    players = state["players"]

    cards_faceup = players["cards_faceup"]
    cards_faceup = cards_faceup.at[target, choice].set(True)

    players |= {"cards_faceup": cards_faceup}
    state |= {"players": players, "discard_choice_queued": jnp.bool(False)}

    return state
