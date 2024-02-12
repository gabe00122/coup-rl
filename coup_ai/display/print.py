from ..gamerules.player import PlayersState
from ..gamerules.deck import DeckState

card_names = ["None", "Assassin", "Duke", "Captain", "Ambassador", "Contessa"]


def print_player_state(cards: list[int], coins: int, cards_faceup: list[bool]):
    card1, card2 = [card_names[card] for card in cards]
    card1_faceup, card2_faceup = cards_faceup

    print(f"  {card1} ({'faceup' if card1_faceup else 'facedown'})")
    print(f"  {card2} ({'faceup' if card2_faceup else 'facedown'})")
    print(f"  Coins: {coins}")
    print()


def print_players_state(players: PlayersState):
    cards = players["cards"].tolist()
    coins = players["coins"].tolist()
    cards_faceup = players["cards_faceup"].tolist()

    for i in range(len(cards)):
        print(f"Player {i + 1}:")
        print_player_state(cards[i], coins[i], cards_faceup[i])


def print_deck(deck: DeckState):
    cards = deck.cards[:deck.size].tolist()
    names = [card_names[card] for card in cards]
    print(f"Deck: {names}")
