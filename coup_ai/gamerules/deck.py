from jax import random
from typing import NamedTuple
from jaxtyping import Array, Scalar, UInt32, UInt8, PRNGKeyArray


class DeckState(NamedTuple):
    cards: UInt8[Array, "15"]
    size: UInt32[Scalar, ""]


def draw_card(deck: DeckState, rng_key: PRNGKeyArray) -> tuple[DeckState, UInt8[Scalar, ""]]:
    draw_index = random.randint(rng_key, (), 0, deck.size)
    card = deck.cards[draw_index]
    new_cards = deck.cards.at[draw_index].set(deck.cards[deck.size - 1])
    new_size = deck.size - 1

    return DeckState(cards=new_cards, size=new_size), card


def replace_card(deck: DeckState, rng_key: PRNGKeyArray, insert: UInt8[Scalar, ""]) -> tuple[DeckState, UInt8[Scalar, ""]]:
    draw_index = random.randint(rng_key, (), 0, deck.size)
    card = deck.cards[draw_index]
    new_cards = deck.cards.at[draw_index].set(insert)

    return DeckState(cards=new_cards, size=deck.size), card


def insert_card(deck: DeckState, card: UInt8[Scalar, ""]) -> DeckState:
    new_cards = deck.cards.at[deck.size].set(card)
    new_size = deck.size + 1

    return DeckState(cards=new_cards, size=new_size)
