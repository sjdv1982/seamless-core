"""Shared validation of flat deep indexes and pin values."""
from seamless.checksum_class import Checksum

DEEP_CELLTYPES = frozenset(("deepcell", "deepfolder", "folder"))

def validate_deep_structure(value, *, index=True):
    if not isinstance(value, dict):
        raise ValueError("A deep structure must be a flat dict")
    result = {}
    for key, member in value.items():
        if not isinstance(key, str):
            raise ValueError(f"Deep key {key!r} must be a string")
        if index:
            if isinstance(member, (dict, list)):
                raise ValueError(
                    f"Deep member {key!r} is nested; a flat structure is required"
                )
            if not isinstance(member, Checksum) and not (
                isinstance(member, str)
                and len(member) == 64
                and all(c in "0123456789abcdef" for c in member)
            ):
                raise ValueError(f"Deep member {key!r} must be a lowercase checksum")
            member = Checksum(member)
        result[key] = member
    return result
