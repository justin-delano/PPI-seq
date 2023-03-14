from typing import Iterable


def str_subs(
    strs: Iterable, subs: str | list[str]  # pylint: disable=E1131
) -> list[str]:
    """Find strings in list containing one of given substrings

    Args:
        strs (Iterable): List of strings (or pandas index)
        subs (str | list[str]): Substrings to match

    Returns:
        list[str]: Strings that contain one of given substrings
    """

    if isinstance(subs, str):
        subs = [subs]
    return [str for str in strs if any(sub in str for sub in subs)]


def ordered_unique(sequence: Iterable) -> list:
    """Make list of unique values preserving order

    Args:
        sequence (Iterable): To be sorted

    Returns:
        list: List of unique values
    """
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def get_matching_replicates(
    general_sample: str, all_samples: Iterable[str], replicates: Iterable[str]
) -> list[str]:
    """Given a general sample, finds all matching experimental replicates in list of all replicates.

    Args:
        general_sample (str): Sample to match against
        all_samples (Iterable[str]): List of all samples
        replicates (Iterable[str]): Names of replicates to match with

    Returns:
        list[str]: Matching columns
    """
    return [
        sample
        for sample in all_samples
        for replicate in replicates
        if sample.replace(replicate, "") == general_sample
    ]
