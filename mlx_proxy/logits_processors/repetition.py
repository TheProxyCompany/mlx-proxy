from collections.abc import Callable

import mlx.core as mx


def make_repetition_penalty(penalty: float, context_size: int = 20) -> Callable[[mx.array, list[int]], mx.array]:
    """
    Make repetition penalty processor.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        penalty (float): The repetition penalty factor to be applied.
        context_size (int): The number of previous tokens to use.
            Default: ``20``.

    Returns:
        Callable[[mx.array, List[int]], mx.array]:
            The repetition penalty processor.
    """
    if penalty < 0 or not isinstance(penalty, (int, float)):
        raise ValueError(f"penalty must be a non-negative float, got {penalty}")

    def repetition_penalty_processor(tokens: mx.array, logits: mx.array) -> mx.array:
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            selected_logits = logits[:, tokens]
            selected_logits = mx.where(
                selected_logits < 0,
                selected_logits * penalty,
                selected_logits / penalty,
                stream=mx.cpu
            )
            logits[:, tokens] = selected_logits
        return logits

    return repetition_penalty_processor
