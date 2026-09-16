#!/usr/bin/env python3
"""Export a PyTorch ZipDepth state_dict without importing PyTorch.

The official .pth file is a standard zip-based torch.save archive. This
restricted unpickler only reconstructs tensor descriptors, then copies their
float32 storage into one contiguous binary plus a small JSON manifest.
"""

import argparse
import collections
import io
import json
import pickle
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StorageType:
    name: str


@dataclass(frozen=True)
class Storage:
    key: str
    dtype: str
    count: int


@dataclass(frozen=True)
class Tensor:
    storage: Storage
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def rebuild_tensor(storage, storage_offset, shape, stride, *unused):
    return Tensor(storage, storage_offset, tuple(shape), tuple(stride))


class TorchStateDictionaryUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return StorageType(name)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"unsupported checkpoint global: {module}.{name}")

    def persistent_load(self, persistent_id):
        kind, storage_type, key, _location, count = persistent_id
        if kind != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent object: {kind}")
        return Storage(key, storage_type.name, count)


def contiguous_bytes(raw_storage, tensor):
    if tensor.storage.dtype != "FloatStorage":
        raise ValueError(f"{tensor.storage.dtype} is not a float tensor")
    element_count = 1
    for dimension in tensor.shape:
        element_count *= dimension
    expected_stride = []
    running_stride = 1
    for dimension in reversed(tensor.shape):
        expected_stride.append(running_stride)
        running_stride *= dimension
    if tuple(reversed(expected_stride)) != tensor.stride:
        raise ValueError(f"non-contiguous tensor with shape {tensor.shape}")
    start = tensor.storage_offset * 4
    end = start + element_count * 4
    return raw_storage[start:end], element_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_binary", type=Path)
    parser.add_argument("output_manifest", type=Path)
    arguments = parser.parse_args()

    with zipfile.ZipFile(arguments.checkpoint) as archive:
        root = archive.namelist()[0].split("/", 1)[0]
        state_dictionary = TorchStateDictionaryUnpickler(
            io.BytesIO(archive.read(f"{root}/data.pkl"))
        ).load()

        if "state_dict" in state_dictionary:
            state_dictionary = state_dictionary["state_dict"]

        manifest = {}
        offset = 0
        with arguments.output_binary.open("wb") as output:
            for original_name, tensor in state_dictionary.items():
                if not isinstance(tensor, Tensor) or tensor.storage.dtype != "FloatStorage":
                    continue
                name = original_name
                while name.startswith("module.") or name.startswith("_orig_mod."):
                    name = name.split(".", 1)[1]
                raw_storage = archive.read(f"{root}/data/{tensor.storage.key}")
                tensor_bytes, element_count = contiguous_bytes(raw_storage, tensor)
                output.write(tensor_bytes)
                manifest[name] = {
                    "offset": offset,
                    "count": element_count,
                    "shape": list(tensor.shape),
                    "dtype": "float32",
                }
                offset += element_count

    arguments.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Exported {len(manifest)} tensors ({offset:,} float32 values)")


if __name__ == "__main__":
    main()
