from typing import TypedDict
import jax
from jax import random, numpy as jnp
from jaxtyping import Array, UInt8, Bool, Scalar, PRNGKeyArray
from .deck import DeckState


# player count is 3-6 but can but the arrays are always the same size
class PlayersState(TypedDict):
    count: UInt8[Scalar, ""]
    coins: UInt8[Array, "6"]
    cards: UInt8[Array, "6 2"]
    cards_faceup: Bool[Array, "6 2"]


def init_player_state(deck: DeckState, player_count: UInt8[Scalar, ""], rng_key: PRNGKeyArray) -> tuple[DeckState, PlayersState]:
    cards_per_player = 2

    cards = random.permutation(rng_key, deck.cards)
    player_cards = jnp.flip(cards[-12:]).reshape((6, cards_per_player))

    deck = DeckState(cards, deck.size - (player_count * cards_per_player))

    return deck, {
        "count": player_count,
        "coins": jnp.full((6,), 2, dtype=jnp.uint8),
        "cards": player_cards,
        "cards_faceup": jnp.full((6, cards_per_player), False, dtype=jnp.bool)
    }
