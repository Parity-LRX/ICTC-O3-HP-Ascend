"""Version-aware caching of fixed species metadata, never learned features."""
import weakref

import torch

_COMPACT = weakref.WeakKeyDictionary()


def validate_species(index, count):
    key = count, index._version
    if getattr(index, "_ictc_ascend_valid_species", None) == key:
        return
    if index.numel():
        bounds = torch.stack((index.amin(), index.amax())).cpu().tolist()
        if bounds[0] < 0 or bounds[1] >= count:
            raise ValueError("Species index is outside the weight bank")
    index._ictc_ascend_valid_species = key


def compact_species(owner, atomic_numbers):
    mapping = owner.atomic_number_to_index
    key = (atomic_numbers._version, id(mapping), mapping._version,
           atomic_numbers.device, mapping.device, owner.num_elements)
    cached = _COMPACT.get(owner)
    if (cached is not None and cached[0]() is atomic_numbers and cached[1] == key
            and cached[2]._version == cached[3]):
        return cached[2]
    if atomic_numbers.numel():
        bounds = torch.stack((atomic_numbers.amin(), atomic_numbers.amax())).cpu().tolist()
        if bounds[0] < 0 or bounds[1] >= mapping.numel():
            raise ValueError("Atomic numbers are outside the model's compact mapping")
    index = mapping[atomic_numbers]
    validate_species(index, owner.num_elements)
    _COMPACT[owner] = weakref.ref(atomic_numbers), key, index, index._version
    return index
