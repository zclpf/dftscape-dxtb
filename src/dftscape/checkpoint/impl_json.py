from __future__ import annotations
from typing import Mapping, Tuple, Dict, Any
import os
import json
from .registry import CheckpointRegistry


@CheckpointRegistry.register("json")
class JSONCheckpointer:

    def __init__(self, indent: int = 2) -> None:
        self.indent = indent

    def save_arrays_and_scalars(
        self,
        path: str,
        arrays: Mapping[str, Any],
        scalars: Mapping[str, Any] | None = None,
    ) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        payload: Dict[str, Any] = {}

        # Convert arrays to lists for JSON serialization
        for k, v in arrays.items():
            if hasattr(v, "tolist"):  # numpy array
                payload[k] = v.tolist()
            else:  # assume it's already list-like
                payload[k] = list(v)

        # Add scalars
        if scalars:
            payload.update(scalars)

        # Save to JSON file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=self.indent, ensure_ascii=False)

    def save(self, path: str, data: Dict[str, Any]) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Convert any numpy arrays to lists for JSON serialization
        processed_data = {}
        for k, v in data.items():
            if hasattr(v, "tolist"):  # numpy array
                processed_data[k] = v.tolist()
            else:
                processed_data[k] = v

        # Save to JSON file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=self.indent, ensure_ascii=False)

    def load(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_arrays_and_scalars(
        self, path: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        arrays: Dict[str, Any] = {}
        scalars: Dict[str, Any] = {}

        for k, v in data.items():
            if isinstance(v, list):
                arrays[k] = v
            else:
                scalars[k] = v

        return arrays, scalars
