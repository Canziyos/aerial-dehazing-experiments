from __future__ import annotations

import argparse

from src.dehazing.architectures.factory import create_model, get_supported_model_names
from src.dehazing.evaluation.metrics import count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print dehazing model information")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=get_supported_model_names(),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = create_model(args.model)

    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)

    print(f"model: {args.model}")
    print(f"trainable_parameters: {trainable_params}")
    print(f"total_parameters: {total_params}")


if __name__ == "__main__":
    main()