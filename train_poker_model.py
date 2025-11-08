import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import treys

from poker_model import PokerModel

# Ensure reproducibility with a seed
torch.manual_seed(0)

LEARN_RATE = 0.003
MAX_EPOCHS = 200
BATCH_SIZE = 65536
PATIENCE = 10  # for early stopping

CARD_SUITS = {"s": 0, "d": 1, "c": 2, "h": 3}
CARD_VALS = {
    **{str(n): n for n in range(2, 10)},
    "T": 10, 
    "J": 11, 
    "Q": 12, 
    "K": 13, 
    "A": 14,
}
CANON_ORDER = ["s", "d", "c", "h"]

evaluator = treys.Evaluator()


def canonicalise_suits(hand_str, board_str):
    """Maps the suits that appear in (hand + board) to a fixed canonical order based 
    on first appearance. Returns new (hand_str, board_str) with suits remapped.
    """
    hand_list = hand_str.split(" ")
    board_list = [] if pd.isna(board_str) else board_str.split(" ")

    suit_map = {}
    next_canon_index = 0

    # Loop through all currently visible cards to build the suit map
    for card in hand_list + board_list:
        suit = card[1]

        if suit not in suit_map:
            suit_map[suit] = CANON_ORDER[next_canon_index]
            next_canon_index += 1

    # Rebuild the hand and board with canonical suits
    hand_list_canon = []
    board_list_canon = []

    for card in hand_list:
        rank = card[0]
        original_suit = card[1]

        new_suit = suit_map[original_suit]
        new_card = rank + new_suit

        hand_list_canon.append(new_card)
    
    for card in board_list:
        rank = card[0]
        original_suit = card[1]

        new_suit = suit_map[original_suit]
        new_card = rank + new_suit

        board_list_canon.append(new_card)
    
    hand_str_canon = " ".join(hand_list_canon)
    board_str_canon = " ".join(board_list_canon)

    return (hand_str_canon, board_str_canon)


def vectorise_cards(hand_str, board_str):
    """Converts hand and board strings into a numerical vector of 1s and 0s
     
    This process involves:
    1) Suit canonicalisation to create a standard representation of the cards.
    2) One-hot encoding of the canonical cards.
    3) Adding extra features such as hand rank to increase model learning speed.

    Total features: 104 + 3 + 9 + 1 + 1 + 1 = 119
    """
    (hand_str, board_str) = canonicalise_suits(hand_str, board_str)

    # Initial list for one-hot encoding of cards
    vector = [0] * 104
    hand_list = hand_str.split(" ")
    board_list = [] if not board_str else board_str.split(" ")

    # Enumerate two segments: 0 = hand(first 52 features), 1 = board (next 52 features)
    for segment, cards in enumerate([hand_list, board_list]):
        offset = 52 * segment  # 0 for hand, 52 for board.
        for card in cards:
            card_val, card_suit = CARD_VALS[card[0]], CARD_SUITS[card[1]]

            # CARD_VALS uses values 2-14, we want indices 0-12
            rank_index = card_val - 2

            one_index = offset + (card_suit*13 + rank_index)
            vector[one_index] = 1
    
    # Additional features

    # One-hot encoding of street (+3 features)
    street_vector = [0] * 3
    board_length_to_street_index_map = {3: 0, 4: 1, 5: 2}
    street_one_index = board_length_to_street_index_map[len(board_list)]
    street_vector[street_one_index] = 1
    vector += street_vector

    # One-hot encoding of hand rank (+9 features)
    hand_rank_vector = [0] * 9

    #Convert card strings to treys objects for compatibility with the evaluator
    hand_ints = [treys.Card.new(card) for card in hand_list]
    board_ints = [treys.Card.new(card) for card in board_list]

    score = evaluator.evaluate(hand_ints, board_ints)
    hand_rank = evaluator.get_rank_class(score)

    # hand_rank is an int from 1-9, so convert it to a 0-based index
    hand_rank_one_index = hand_rank - 1
    hand_rank_vector[hand_rank_one_index] = 1
    vector += hand_rank_vector


    # Hand pair feature (1 if hand is a pair, 0 otherwise)
    hand_pair_feature = int(hand_list[0][0] == hand_list[1][0])
    vector.append(hand_pair_feature)

    # Hand suited feature (1 if hand is suited, 0 otherwise)
    hand_suited_feature = int(hand_list[0][1] == hand_list[1][1])
    vector.append(hand_suited_feature)

    # Hand rank gap feature (normalised difference between ranks of the two hand cards)
    val1, val2 = CARD_VALS[hand_list[0][0]], CARD_VALS[hand_list[1][0]]

    # Handle the special case of an Ace, which can be high (14) or low (1)
    if val1 == 14:  # first card is an ace
        diff = min(abs(14 - val2), abs(1 - val2))   
    elif val2 == 14:  # second card is an ace
        diff = min(abs(14 - val1), abs(1 - val1))
    else:  # no ace in hand
        diff = abs(val1 - val2)
    
    # Normalise by dividing by the largest possible gap
    hand_rank_gap_feature = diff / 12
    vector.append(hand_rank_gap_feature)

    return vector


def main():
    """Loads the poker datasets, trains the model, and saves the best version."""
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load training data and format it
    df_training_data = pd.read_csv("training_data.csv")
    training_inputs = df_training_data.apply(
        lambda row: vectorise_cards(row["hand_str"], row["board_str"]), 
        axis=1
    )
    X_training = torch.tensor(training_inputs.tolist(), dtype=torch.float32)
    Y_training = torch.tensor(
        df_training_data["hand_outcome"].values, dtype=torch.float32
    ).unsqueeze(dim=1)

    training_dataset = TensorDataset(X_training, Y_training)

    training_data_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True  #shuffle training data to improve generalisation
    )

    # Load the validation data and format it
    df_validation_data = pd.read_csv("validation_data.csv")
    validation_inputs = df_validation_data.apply(
        lambda row: vectorise_cards(row["hand_str"], row["board_str"]), axis=1
    )
    X_validation = torch.tensor(validation_inputs.tolist(), dtype=torch.float32)
    Y_validation = torch.tensor(
        df_validation_data["hand_equity"].values, dtype=torch.float32
    ).unsqueeze(dim=1)

    # Instantiate the model, loss function, and optimiser
    model = PokerModel()
    model.to(device)

    loss_function = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARN_RATE)

    # Training loop
    print("Starting training")

    best_validation_mae = float("inf") 
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()

        for batch_inputs, batch_labels in training_data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            prediction_logits = model(batch_inputs)

            # Compare raw logits to true labels
            loss = loss_function(prediction_logits, batch_labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        model.eval()

        with torch.no_grad():  # for efficiency, as gradients are not needed
            X_validation = X_validation.to(device)
            Y_validation = Y_validation.to(device)

            validation_logits = model(X_validation)

            # Convert raw logits to probabilities to match Y_validation
            validation_probs = torch.sigmoid(validation_logits)

            validation_errors = torch.abs(validation_probs - Y_validation)
            validation_mae = torch.mean(validation_errors).item()
        

        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch+1} complete. validation_mae: {validation_mae:.5f}")

        if epochs_without_improvement == PATIENCE:
            break

        

    # Load the best version of the model and save to file
    model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), "poker_model_state.pth")

    print(f"Training complete. Best validation_mae: {best_validation_mae:.5f}")


if __name__ == "__main__":
    main()

