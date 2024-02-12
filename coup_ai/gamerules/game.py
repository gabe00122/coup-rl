from jax import random, numpy as jnp
from typing import TypedDict
from jaxtyping import Scalar, UInt8, Bool, PRNGKeyArray
from .player import PlayersState, init_player_state
from .deck import DeckState, draw_card

# phases
action_phase = jnp.uint8(0)
challenge_phase = jnp.uint8(1)
challenge_lose_card_phase = jnp.uint8(2)
exchange_phase = jnp.uint8(3)
block_phase = jnp.uint8(4)
challenge_block_phase = jnp.uint8(5)
challenge_block_lose_card_phase = jnp.uint8(6)
action_lose_card_phase = jnp.uint8(7)


class GameState(TypedDict):
    players: PlayersState
    deck: DeckState
    active_player: UInt8[Scalar, ""]

    phase: UInt8[Scalar, ""]
    # phase: Action
    action_alive: Bool[Scalar, ""]  # is the action going to be preformed if it's not stopped?
    action: UInt8[Scalar, ""]
    target: UInt8[Scalar, ""]  # the target of the action

    # phase: Block
    block_alive: Bool[Scalar, ""]

    # lose card
    discard_choice_queued: Bool[Scalar, ""]
    discard_choice_target: UInt8[Scalar, ""]  # the target of the discard choice


def initial_deck() -> DeckState:
    card_types = 5
    card_instances = 3
    cards = jnp.array([i + 1 for i in range(card_types) for _ in range(card_instances)])
    return DeckState(cards=cards, size=jnp.uint32(card_types * card_instances))


def initial_state(player_count: UInt8[Scalar, ""], rng_key: PRNGKeyArray) -> GameState:
    deck = initial_deck()
    deck, players = init_player_state(deck, player_count, rng_key)

    return GameState(
        players=players,
        deck=deck,
        active_player=jnp.uint8(0),
        phase=jnp.uint8(0),
        action_alive=jnp.bool(False),
        action=jnp.uint8(0),
        target=jnp.uint8(0),
        block_alive=jnp.bool(False),
        discard_choice_queued=jnp.bool(False),
        discard_choice_target=jnp.uint8(0),
    )


def end_turn(state: GameState) -> GameState:
    return state | {
        "active_player": (state["active_player"] + 1) % len(state["players"]),
        "phase": jnp.uint8(0),
        "action": jnp.uint8(0),
        "target": jnp.uint8(0),
        # "discard_choice_queued": jnp.bool_(False),
    }
