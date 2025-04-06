import mlx.core as mx


def calculate_entropy(
    logprobs: mx.array,
    epsilon: float = 1e-9
) -> mx.array:
    """
    Calculate the entropy and normalized entropy of the given logprobs.

    Args:
        logprobs (mx.array): Array of logprobs. Can be 1D or 2D (for batched inputs).
        epsilon (float): Small value to avoid numerical instability. Default is 1e-9.

    Returns:
        tuple[mx.array, mx.array]: A tuple containing:
        - The entropy.
        - The normalized entropy.
    """
    # Ensure logprobs is 2D for consistent processing
    if logprobs.ndim == 1:
        logprobs = logprobs.reshape(1, -1)

    num_classes = logprobs.shape[-1]
    max_entropy = -mx.log(1.0 / num_classes)
    if max_entropy <= 0.0:
        max_entropy = 1.0  # Safeguard

    # Calculate probs from logprobs
    probs = mx.exp(logprobs, stream=mx.cpu)  # p_i = exp(log(p_i))

    # Clip probs for stability (optional but often good)
    probs = mx.clip(probs, epsilon, 1.0 - epsilon, stream=mx.cpu)
    # Re-normalize clipped probs? Or rely on clipping logprobs earlier? Careful here.

    # Calculate entropy using probs and logprobs
    # H = - sum(p_i * log(p_i)) = - sum(exp(logprobs_i) * logprobs_i)
    entropy = mx.sum(-probs * logprobs, axis=-1, stream=mx.cpu)  # Use the input logprobs here

    normalized_entropy = entropy / max_entropy

    # Double-check for NaN values in entropy and normalized_entropy
    assert not mx.isnan(entropy), "Entropy calculation resulted in NaN"
    assert not mx.isnan(normalized_entropy), (
        "Normalized entropy calculation resulted in NaN"
    )

    return entropy, normalized_entropy
