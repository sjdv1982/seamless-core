"""API names that must never fall through to structural navigation.

Entries are activated when the corresponding API is removed. Item navigation
continues to permit value keys with these names.
"""

RETIRED_NAMES: dict[str, str] = {"target_celltype": "celltype"}


def check_retired_name(name: str) -> None:
    replacement = RETIRED_NAMES.get(name)
    if replacement is not None:
        raise AttributeError(f"{name!r} has been retired; use {replacement} instead")
