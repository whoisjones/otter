from torch.utils.data import WeightedRandomSampler


def language_of(example_id):
    # Training files are per-language and their ids carry the language as a prefix:
    # "<lang>_<document>-s<sentence>", e.g. "tha_2200-s0". An id in any other shape has
    # no language and keeps the default weight.
    if not isinstance(example_id, str) or "_" not in example_id:
        return None
    return example_id.split("_", 1)[0]


def language_counts(dataset):
    if "id" not in dataset.column_names:
        raise ValueError(
            "language_weights needs an 'id' column to read the language from; "
            f"this dataset has {dataset.column_names}."
        )
    counts = {}
    for example_id in dataset["id"]:
        language = language_of(example_id)
        counts[language] = counts.get(language, 0) + 1
    return counts


def build_language_sampler(dataset, language_weights, generator=None):
    # Draws the corpus in the proportions the weights ask for instead of uniformly.
    # A weight is a multiplier on a language's existing mass, so a language holding 2%
    # of the sentences at weight 5 ends up near 10% of the draws -- `language_mixture`
    # reports the exact shares. Sampling is with replacement, so an upweighted language
    # repeats within an epoch rather than the others being dropped.
    counts = language_counts(dataset)
    unknown = sorted(set(language_weights) - set(counts))
    if unknown:
        raise ValueError(
            f"language_weights names languages absent from the training data: {unknown}. "
            f"Present: {sorted(language for language in counts if language)}"
        )
    weights = [
        float(language_weights.get(language_of(example_id), 1.0)) for example_id in dataset["id"]
    ]
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True, generator=generator
    )


def language_mixture(dataset, language_weights):
    # The share of draws each language receives under those weights.
    counts = language_counts(dataset)
    mass = {
        language: count * float(language_weights.get(language, 1.0))
        for language, count in counts.items()
    }
    total = sum(mass.values())
    return {
        language: value / total
        for language, value in sorted(mass.items(), key=lambda x: x[0] or "")
    }
