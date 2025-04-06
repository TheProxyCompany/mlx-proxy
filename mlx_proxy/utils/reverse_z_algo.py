from typing import Sequence, TypeVar

T = TypeVar("T")


def reverse_z_algorithm(
    last_elements: Sequence[T],
    sequence_breakers: dict[T, list[list[T]]],
    last_n: int | None = None,
    repetition_limit: int = -1,
) -> list[int]:
    """
    Implements a reverse Z-algorithm to efficiently compute the positions and lengths of suffixes
    appearing elsewhere in the context, while considering sequence breakers and an optional repetition limit.

    This function identifies repeated sequences in the context to apply penalties and prevent looping
    in text generation. It's a novel fork of the traditional Z-algorithm, adapted for reverse processing
    and incorporating sequence breakers and an optional repetition limit.

    Args:
        last_elements (Sequence[T]): The sequence of elements to analyze, typically the last N elements of the context.
        sequence_breakers (Dict[T, List[List[T]]]): The sequence breakers, mapping head tokens to tail sequences.
        last_n (Optional[int]): The number of elements to consider. If None, uses the full length of last_elements.
        repetition_limit (int): The maximum length of repetition to consider. If -1, no limit is applied.

    Returns:
        List[int]: A list of repeat counts for each position in the input sequence.
                   Each count represents the length of the longest suffix at that position
                   that also appears elsewhere in the sequence.

    Example:
        >>> elements = [1, 2, 3, 3, 2, 3, 4, 1, 2, 3]
        >>> reverse_z_algorithm(elements, sequence_breakers={})
        [0, 0, 3, 1, 0, 2, 0, 0, 0, 0]
    """
    if not last_elements:
        return []

    if last_n is None:
        last_n = len(last_elements)

    n = last_n
    repeat_counts = [0] * n
    # Initialize z_box boundaries
    z_box_start = n - 1  # Start index of the current Z-box
    z_box_end = n - 1  # End index of the current Z-box

    # Process the sequence in reverse order
    for k in range(n - 2, -1, -1):
        token = last_elements[k]
        # Check if this token is part of a sequence breaker
        is_breaker, _ = is_sequence_breaker(token, sequence_breakers, last_elements, k)
        if is_breaker:
            continue

        if k < z_box_start:
            # Case when k is outside the current Z-box
            n_match = 0
            while (k - n_match >= 0) and (
                last_elements[k - n_match] == last_elements[n - 1 - n_match]
            ):
                # Check for sequence breaker
                t = last_elements[k - n_match]
                is_breaker, _ = is_sequence_breaker(
                    t, sequence_breakers, last_elements, k - n_match
                )
                if is_breaker:
                    break
                n_match += 1
                if repetition_limit != -1 and n_match >= repetition_limit:
                    break
            repeat_counts[k] = n_match
            if n_match > 0:
                z_box_start = k - n_match + 1
                z_box_end = k
        else:
            # Case when k is inside the current Z-box
            k_prime = n - 1 - (z_box_end - k)
            beta_length = z_box_end - k + 1
            if repeat_counts[k_prime] < beta_length:
                # Case 1: Value is within the Z-box
                repeat_counts[k] = repeat_counts[k_prime]
            else:
                # Case 2: Need to check beyond the Z-box
                n_match = beta_length
                while (k - n_match >= 0) and (
                    last_elements[k - n_match] == last_elements[n - 1 - n_match]
                ):
                    # Check for sequence breaker
                    t = last_elements[k - n_match]
                    is_breaker, _ = is_sequence_breaker(
                        t, sequence_breakers, last_elements, k - n_match
                    )
                    if is_breaker:
                        break
                    n_match += 1
                    if repetition_limit != -1 and n_match >= repetition_limit:
                        break
                repeat_counts[k] = n_match
                z_box_start = k - n_match + 1
                z_box_end = k

    return repeat_counts


def is_sequence_breaker(
    token: T,
    sequence_breakers: dict[T, list[list[T]]],
    context: Sequence[T],
    position: int,
) -> tuple[bool, int]:
    """
    Checks if the given token (at the specified position) is part of a sequence breaker.

    Sequence breakers are tokens or token sequences that interrupt repetition counting.

    Args:
        token (T): The current token being processed.
        sequence_breakers (Dict[T, List[List[T]]]): The sequence breakers mapping head tokens to tail sequences.
        context (Sequence[T]): The context being checked for repetition.
        position (int): The position of the token in the context.

    Returns:
        Tuple[bool, int]: A tuple where:
            - The first element is True if the token and its subsequent tokens form a sequence breaker, False otherwise.
            - The second element is the length of the sequence breaker if it is a sequence breaker, -1 otherwise.
    """
    if token not in sequence_breakers:
        return False, -1

    possible_sequences = sequence_breakers[token]

    for seq in possible_sequences:
        seq_len = len(seq)
        if seq_len == 0:  # Single token sequence breaker
            return True, 1
        available_length = len(context) - (position + 1)
        if (
            available_length >= seq_len
            and context[position + 1 : position + 1 + seq_len] == seq
        ):
            return (
                True,
                seq_len + 1,
            )  # Include head token in the sequence breaker length

    return False, -1
