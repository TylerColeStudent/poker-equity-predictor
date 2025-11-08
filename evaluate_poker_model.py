import pandas as pd
import torch

from poker_model import PokerModel
from train_poker_model import vectorise_cards


def main():
    """Evaluates the performance of a trained poker model with an unseen test dataset.

    Calculates the overall Mean Absolute Error (MAE) along with the MAE for each street,
    and a baseline MAE from consistently guessing 0.5 for comparison.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data and format it
    df_test_data = pd.read_csv("test_data.csv")

    test_inputs = df_test_data.apply(
        lambda row: vectorise_cards(row["hand_str"], row["board_str"]), 
        axis=1
    )
    X_test = torch.tensor(test_inputs.tolist(), dtype=torch.float32)
    Y_test = torch.tensor(
        df_test_data["hand_equity"].values, dtype=torch.float32
    ).unsqueeze(dim=1)
    
    # Load the best model version picked by validation
    model = PokerModel()
    model.load_state_dict(torch.load("poker_model_state.pth"))
    model.to(device)

    model.eval()

    with torch.no_grad():  # for efficiency, as gradients are not needed
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)

        prediction_logits = model(X_test)
        prediction_probs = torch.sigmoid(prediction_logits)

        test_errors = torch.abs(prediction_probs - Y_test)
        test_mae = torch.mean(test_errors).item()

    print(f"Overall test MAE: {test_mae:.5f}")

    # Compute the 95% confidence interval
    test_std = torch.std(test_errors)
    error_margin = 1.96 * test_std / (len(Y_test) ** 0.5)
    lower_bound = test_mae - error_margin
    upper_bound = test_mae + error_margin
    print(f"95% confidence interval: [{lower_bound:.6f}, {upper_bound:.6f}]\n")

    # Compute the MAE for each street
    board_lengths = df_test_data["board_str"].apply(lambda str: len(str.split()))

    flop_indices = board_lengths[board_lengths == 3].index
    turn_indices = board_lengths[board_lengths == 4].index
    river_indices = board_lengths[board_lengths == 5].index

    flop_errors = test_errors[flop_indices]
    turn_errors = test_errors[turn_indices]
    river_errors = test_errors[river_indices]

    flop_mae = torch.mean(flop_errors).item()
    turn_mae = torch.mean(turn_errors).item()
    river_mae = torch.mean(river_errors).item()

    print(f"Flop MAE: {flop_mae:.5f}")
    print(f"Turn MAE: {turn_mae:.5f}")
    print(f"River MAE: {river_mae:.5f}\n")

    # Compute the MAE for consistently guessing 0.5 for a basic benchmark
    trivial_predictions = torch.full_like(Y_test, 0.5)
    trivial_errors = torch.abs(trivial_predictions - Y_test)
    trivial_mae = torch.mean(trivial_errors).item()
    print(f"Trivial 0.5 guess MAE: {trivial_mae:.5f}")


if __name__ == "__main__":
    main()


# BEST SO FAR = 0.0202 w/ weight_decay = 0.01, lr = 0.002, and perhaps 128-64-16 model arch