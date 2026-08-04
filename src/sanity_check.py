"""Sanity check wiring preprocessing -> Dataset -> LSTM model together.

This does not train anything; it only verifies that tensor shapes flow
correctly through the pipeline before the training stage is implemented.
"""

from dataset import create_dataloaders
from model import LSTMRegressor
from preprocessing import prepare_data


def run_sanity_check(
    symbol: str = "MSFT", sequence_length: int = 60, batch_size: int = 32
) -> None:
    data = prepare_data(symbol, sequence_length=sequence_length)
    train_loader, _, _ = create_dataloaders(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
        batch_size=batch_size,
    )

    X_batch, y_batch = next(iter(train_loader))
    expected_batch_size = min(batch_size, len(data["X_train"]))

    assert X_batch.shape == (expected_batch_size, sequence_length, 1), (
        f"Unexpected X batch shape: {X_batch.shape}"
    )
    assert y_batch.shape == (expected_batch_size, 1), (
        f"Unexpected y batch shape: {y_batch.shape}"
    )

    model = LSTMRegressor()
    output = model(X_batch)

    assert output.shape == (expected_batch_size, 1), (
        f"Unexpected model output shape: {output.shape}"
    )

    print(f"X batch shape: {X_batch.shape}")
    print(f"y batch shape: {y_batch.shape}")
    print(f"Model output shape: {output.shape}")
    print("Sanity check passed.")


if __name__ == "__main__":
    run_sanity_check()
