# The cross-encoder prepends a label prefix of `label_offset` characters. A token
# belongs to the text once its END offset passes the prefix -- testing the START
# offset instead skips the first text token whenever the tokenizer folds the
# preceding whitespace into it (mmBERT and BPE tokenizers generally), which makes
# the first word unreachable for both gold labels and predictions.
#
# The minimum over the batch is taken because the token straddling the
# prefix/text boundary can differ per sample; a per-sample index would produce
# mismatched tensor shapes when the batch is stacked.
def first_text_token_index(offset_mapping, label_offset):
    indices = []
    for offsets in offset_mapping:
        for idx in range(len(offsets)):
            if int(offsets[idx][1]) > label_offset:
                indices.append(idx)
                break
    return min(indices) if indices else 0


def all_spans_mask(input_ids, sequence_ids, max_span_length):
    text_start_index = 0
    while sequence_ids[text_start_index] is None:
        text_start_index += 1

    text_end_index = len(input_ids) - 1
    while sequence_ids[text_end_index] is None:
        text_end_index -= 1

    start_mask, end_mask = [], []
    for sequence_id in sequence_ids:
        start_mask.append(1 if sequence_id is not None else 0)
        end_mask.append(1 if sequence_id is not None else 0)

    span_mask = [
        [(j - i >= 0 and j - i < max_span_length) * s * e for j, e in enumerate(end_mask)]
        for i, s in enumerate(start_mask)
    ]

    return text_start_index, text_end_index, start_mask, end_mask, span_mask


def compressed_all_spans_mask(input_ids, sequence_ids, max_span_length):
    num_tokens = len(input_ids)
    text_start_index = 0
    while sequence_ids[text_start_index] is None:
        text_start_index += 1

    text_end_index = len(input_ids) - 1
    while sequence_ids[text_end_index] is None:
        text_end_index -= 1

    start_mask, end_mask = [], []
    for sequence_id in sequence_ids:
        start_mask.append(1 if sequence_id is not None else 0)
        end_mask.append(1 if sequence_id is not None else 0)

    spans_idx = [
        (i, i + j) for i in range(num_tokens) for j in range(max_span_length) if i + j < num_tokens
    ]
    span_lengths = [
        end_subword_index - start_subword_index + 1
        for start_subword_index, end_subword_index in spans_idx
    ]
    span_mask = []

    for start_subword_index, end_subword_index in spans_idx:
        if (
            start_subword_index >= text_start_index
            and end_subword_index <= text_end_index
            and start_mask[start_subword_index] == 1
            and end_mask[end_subword_index] == 1
        ):
            span_mask.append(1)
        else:
            span_mask.append(0)

    return (
        text_start_index,
        text_end_index,
        start_mask,
        end_mask,
        span_mask,
        spans_idx,
        span_lengths,
    )


def subwords_mask(input_ids, word_ids, max_span_length):
    text_start_index = 0
    while word_ids[text_start_index] is None:
        text_start_index += 1

    text_end_index = len(input_ids) - 1
    while word_ids[text_end_index] is None:
        text_end_index -= 1

    start_mask, end_mask = [], []
    previous_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            start_mask.append(1 if word_id != previous_word_id else 0)
            end_mask.append(1 if (idx + 1 == len(word_ids) or word_ids[idx + 1] != word_id) else 0)
        else:
            start_mask.append(0)
            end_mask.append(0)
        previous_word_id = word_id

    span_mask = [
        [(j - i >= 0 and j - i < max_span_length) * s * e for j, e in enumerate(end_mask)]
        for i, s in enumerate(start_mask)
    ]

    return text_start_index, text_end_index, start_mask, end_mask, span_mask


def compressed_subwords_mask(input_ids, word_ids, max_span_length):
    num_tokens = len(input_ids)
    text_start_index = 0
    while text_start_index < num_tokens and word_ids[text_start_index] is None:
        text_start_index += 1

    text_end_index = num_tokens - 1
    while text_end_index >= 0 and word_ids[text_end_index] is None:
        text_end_index -= 1

    start_mask = []
    end_mask = []
    previous_word_id = None

    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            is_start = 1 if word_id != previous_word_id else 0

            if idx + 1 == num_tokens or word_ids[idx + 1] != word_id:
                is_end = 1
            else:
                is_end = 0
        else:
            is_start = 0
            is_end = 0

        start_mask.append(is_start)
        end_mask.append(is_end)
        previous_word_id = word_id

    spans_idx = [
        (i, i + j) for i in range(num_tokens) for j in range(max_span_length) if i + j < num_tokens
    ]
    span_lengths = [
        end_subword_index - start_subword_index + 1
        for start_subword_index, end_subword_index in spans_idx
    ]
    span_mask = []

    for start_subword_index, end_subword_index in spans_idx:
        if (
            start_subword_index >= text_start_index
            and end_subword_index <= text_end_index
            and start_mask[start_subword_index] == 1
            and end_mask[end_subword_index] == 1
        ):
            span_mask.append(1)
        else:
            span_mask.append(0)

    return (
        text_start_index,
        text_end_index,
        start_mask,
        end_mask,
        span_mask,
        spans_idx,
        span_lengths,
    )


def compressed_all_spans_mask_cross_encoder(
    input_ids,
    sequence_ids,
    max_span_length,
    label_offset,
    offsets,
    has_threshold_token=False,
    text_start_index=None,
):
    num_tokens = len(input_ids)

    # Compare the token's END offset against the prefix length, not its start.
    # Tokenizers that fold the preceding whitespace into the token (mmBERT, and
    # BPE tokenizers generally) give the first text token a char-start of
    # label_offset - 1, so a `start < label_offset` test skips it and makes the
    # first word of the text unreachable for both labels and predictions.
    #
    # The caller passes a batch-wide text_start_index (see first_text_token_index):
    # the label prefix is identical for every sample, but the token straddling the
    # prefix/text boundary is not, so computing this per sample yields different
    # values within one batch and torch.stack fails on mismatched shapes.
    if text_start_index is None:
        text_start_index = 0
        while text_start_index < num_tokens and (
            offsets[text_start_index][1] <= label_offset or sequence_ids[text_start_index] is None
        ):
            text_start_index += 1

    text_end_index = len(input_ids) - 1
    while sequence_ids[text_end_index] is None:
        text_end_index -= 1

    start_mask, end_mask = [], []
    if has_threshold_token:
        start_mask.append(0)
        end_mask.append(0)
        max_span_length += 1

    for sequence_id in sequence_ids[text_start_index:]:
        start_mask.append(1 if sequence_id is not None else 0)
        end_mask.append(1 if sequence_id is not None else 0)

    spans_idx = [
        (i, i + j)
        for i in range(num_tokens - text_start_index)
        for j in range(max_span_length)
        if i + j < num_tokens - text_start_index
    ]
    span_lengths = [
        end_subword_index - start_subword_index + 1
        for start_subword_index, end_subword_index in spans_idx
    ]
    span_mask = []

    for start_subword_index, end_subword_index in spans_idx:
        if (
            start_subword_index + text_start_index >= text_start_index
            and end_subword_index + text_start_index <= text_end_index
            and start_mask[start_subword_index] == 1
            and end_mask[end_subword_index] == 1
        ):
            span_mask.append(1)
        else:
            span_mask.append(0)

    return (
        text_start_index,
        text_end_index,
        start_mask,
        end_mask,
        span_mask,
        spans_idx,
        span_lengths,
    )


def compressed_subwords_mask_cross_encoder(
    input_ids,
    word_ids,
    max_span_length,
    label_offset,
    has_threshold_token=False,
    text_start_index=None,
):
    num_tokens = len(input_ids)
    if text_start_index is None:
        text_start_index = 0
        while text_start_index < num_tokens:
            wid = word_ids[text_start_index]
            if wid is None or wid < label_offset:
                text_start_index += 1
                continue
            break

    text_end_index = num_tokens - 1
    while text_end_index >= 0 and word_ids[text_end_index] is None:
        text_end_index -= 1

    start_mask = []
    end_mask = []
    if has_threshold_token:
        start_mask.append(0)
        end_mask.append(0)
        max_span_length += 1

    previous_word_id = None

    text_word_ids = word_ids[text_start_index:]
    for idx, word_id in enumerate(text_word_ids):
        if word_id is not None:
            is_start = 1 if word_id != previous_word_id else 0

            if idx + 1 == len(text_word_ids) or text_word_ids[idx + 1] != word_id:
                is_end = 1
            else:
                is_end = 0
        else:
            is_start = 0
            is_end = 0

        start_mask.append(is_start)
        end_mask.append(is_end)
        previous_word_id = word_id

    spans_idx = [
        (i, i + j)
        for i in range(num_tokens - text_start_index)
        for j in range(max_span_length)
        if i + j < num_tokens - text_start_index
    ]
    span_lengths = [
        end_subword_index - start_subword_index + 1
        for start_subword_index, end_subword_index in spans_idx
    ]
    span_mask = []

    for start_subword_index, end_subword_index in spans_idx:
        if (
            start_subword_index + text_start_index >= text_start_index
            and end_subword_index + text_start_index <= text_end_index
            and start_mask[start_subword_index] == 1
            and end_mask[end_subword_index] == 1
        ):
            span_mask.append(1)
        else:
            span_mask.append(0)

    return (
        text_start_index,
        text_end_index,
        start_mask,
        end_mask,
        span_mask,
        spans_idx,
        span_lengths,
    )
