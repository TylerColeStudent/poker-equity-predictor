import random
from itertools import combinations
from math import comb

import pandas as pd
import treys

# Ensure reproducibility with a seed.
random.seed(0)

TRAIN_HANDS = 10**7
VAL_HANDS = 10**4
TEST_HANDS = 10**4

SIMS_PER_HAND_FLOP = 10**5  # brute force not possible, this gives an estimate
SIMS_PER_HAND_TURN = 46 * comb(45, 2)  # brute force for exact equity
SIMS_PER_HAND_RIVER = comb(45, 2)  # brute force for exact equity

BOARD_LENGTHS = [3, 4, 5]  # post-flop only

evaluator = treys.Evaluator()


def simulate_hand(hand, board, row_index, seed_offset=0):
    """Simulates a given hand and board state to showdown against a random hand. 
    
    Returns 1 if the given hand wins, 0 if it loses, and 0.5 for a tie.
    """

    # Use a unique seed for each hand to ensure reproducible results
    random.seed(row_index + seed_offset)

    # Print the row number every 100000th row to monitor progress during runtime
    if row_index % 10**5 == 0 and row_index > 0:
        print(f"Row {row_index} complete.")

    deck = treys.Deck()

    # Remove known cards to avoid duplicates later
    for card in hand + board:
        deck.cards.remove(card)

    opp_hand = deck.draw(2)
    full_board = board + deck.draw(5-len(board))

    # Note: lower score is better
    score = evaluator.evaluate(hand, full_board)
    opp_score = evaluator.evaluate(opp_hand, full_board)

    if score < opp_score:
        return 1
    elif score > opp_score:
        return 0
    else:
        return 0.5


def get_equity(hand, board, row_index, seed_offset=0):
    """Returns the equity of a given hand and board state.

    Uses brute force for turn and river to get exact win rates.
    Uses Monte Carlo simulation for flop to get accurate estimates for win rates.
    """ 
    
    # Use a unique seed for each hand to ensure reproducible results
    random.seed(row_index + seed_offset)

    # Print the row number every 100th row to monitor progress during runtime
    if row_index % 100 == 0 and row_index > 0:
        print(f"Row {row_index} complete.")
    
    deck = treys.Deck()

    # Remove known cards to avoid duplicates later
    for card in hand + board:
        deck.cards.remove(card)

    available_cards = deck.cards
    equity_sum = 0

    # Monte Carlo for flop
    if len(board) == 3:
        # Flop - opponent hand and two community cards are unknown.
        for _ in range(SIMS_PER_HAND_FLOP):
            random_card_ints = random.sample(available_cards, 4)
            opp_hand = random_card_ints[:2]
            full_board = board + random_card_ints[2:]

            score = evaluator.evaluate(hand, full_board)
            opp_score = evaluator.evaluate(opp_hand, full_board)

            if score < opp_score:
                equity_sum += 1
            elif score == opp_score:
                equity_sum += 0.5
        return equity_sum / SIMS_PER_HAND_FLOP

    # Brute force for turn/river
    if len(board) == 4:
        # Turn - opponent hand and final community card is unknown
        for unknown_cards in combinations(available_cards, 3):
            for river_card in unknown_cards:
                full_board = board + [river_card]
                opp_hand = [card for card in unknown_cards if card != river_card]

                score = evaluator.evaluate(hand, full_board)
                opp_score = evaluator.evaluate(opp_hand, full_board)

                if score < opp_score:
                    equity_sum += 1
                elif score == opp_score:
                    equity_sum += 0.5
        return equity_sum / SIMS_PER_HAND_TURN    

    if len(board) == 5:
        # River - only opponent hand is unknown.
        for opp_hand in combinations(available_cards, 2):
            score = evaluator.evaluate(hand, board) 
            opp_score = evaluator.evaluate(list(opp_hand), board)

            if score < opp_score:
                equity_sum += 1
            elif score == opp_score:
                equity_sum += 0.5
        return equity_sum / SIMS_PER_HAND_RIVER
    

def generate_inputs(num_of_rows):
    """Generates a pandas DataFrame of hand and board inputs, with a given  
    number of rows.
    """
    input_data = []

    for _ in range(num_of_rows):
        deck = treys.Deck()
        hand = deck.draw(2)
        hand_str = " ".join(treys.Card.int_to_str(card) for card in hand)
        board_length = random.choice(BOARD_LENGTHS)
        board = deck.draw(board_length)
        board_str = " ".join(treys.Card.int_to_str(card) for card in board)

        data = {
            "hand": hand, 
            "board": board, 
            "hand_str": hand_str, 
            "board_str": board_str
        }
        input_data.append(data)
    return pd.DataFrame(input_data)


def main():
    """Generates training, validation, and test datasets for a poker neural network

    Creates cheap binary outcome labels for the training dataset, and accurate equity
    labels for the validation and test datasets. Saves the datasets to CSV files.
    """
    
    # Generate the training data 
    print("Generating training data inputs...")
    df_training_data = generate_inputs(TRAIN_HANDS)

    print("Generating training data labels...")
    df_training_data["hand_outcome"] = df_training_data.apply(
        lambda row: simulate_hand(row["hand"], row["board"], row.name),
        axis=1
    )
    print("Training data generation complete.")
    df_training_output = df_training_data[["hand_str", "board_str", "hand_outcome"]]
    df_training_output.to_csv("training_data.csv")

    # Generate the validation data
    print("Generating validation data inputs...")
    seed_offset = TRAIN_HANDS + 1  # for ensuring different seeds to training data
    df_validation_data = generate_inputs(VAL_HANDS)

    print("Generating validation data labels...")
    df_validation_data["hand_equity"] = df_validation_data.apply(
        lambda row: get_equity(row["hand"], row["board"], row.name, seed_offset), 
        axis=1
    )
    print("Validation data generation complete.")
    df_validation_output = df_validation_data[["hand_str", "board_str", "hand_equity"]]
    df_validation_output.to_csv("validation_data.csv")

    # Generate the test data
    print("Generating test data inputs...")
    seed_offset += VAL_HANDS
    df_test_data = generate_inputs(TEST_HANDS)

    print("Generating test data labels...")
    df_test_data["hand_equity"] = df_test_data.apply(
        lambda row: get_equity(row["hand"], row["board"], row.name, seed_offset), 
        axis=1
    )
    print("Test data generation complete.")
    df_test_output = df_test_data[["hand_str", "board_str", "hand_equity"]]
    df_test_output.to_csv("test_data.csv")


if __name__ == "__main__":
    main()


