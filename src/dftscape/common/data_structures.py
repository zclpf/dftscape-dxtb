from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Type, TypeVar
import numpy as np
import json

T = TypeVar("T", bound="SerialisableDataClass")


@dataclass(frozen=True)
class SerialisableDataClass:

    def to_config(self) -> Dict[str, Any]:
        config = {}

        for field_name, field_value in self.__dict__.items():
            if field_value is None:
                continue  # Skip None values
            elif isinstance(field_value, np.ndarray):
                # Convert numpy array to list and store dtype info
                config[field_name] = field_value.tolist()
                config[f"{field_name}_dtype"] = str(field_value.dtype)
            elif isinstance(field_value, (np.integer, np.floating)):
                # Convert numpy scalars to Python types
                config[field_name] = field_value.item()
            elif isinstance(field_value, SerialisableDataClass):
                # Recursively serialize nested SerialisableDataClass objects
                config[field_name] = field_value.to_config()
            elif isinstance(field_value, list):
                # Handle lists that might contain SerialisableDataClass objects
                config[field_name] = []
                for item in field_value:
                    if isinstance(item, SerialisableDataClass):
                        config[field_name].append(item.to_config())
                    elif isinstance(item, tuple):
                        # Handle tuples that might contain SerialisableDataClass objects
                        config[field_name].append(
                            tuple(
                                (
                                    subitem.to_config()
                                    if isinstance(subitem, SerialisableDataClass)
                                    else subitem
                                )
                                for subitem in item
                            )
                        )
                    else:
                        config[field_name].append(item)
            elif isinstance(field_value, dict):
                # Handle dictionaries that might contain SerialisableDataClass objects
                config[field_name] = {}
                for key, value in field_value.items():
                    if isinstance(value, SerialisableDataClass):
                        config[field_name][key] = value.to_config()
                    elif isinstance(value, list):
                        # Handle nested lists in dictionaries
                        config[field_name][key] = [
                            (
                                item.to_config()
                                if isinstance(item, SerialisableDataClass)
                                else item
                            )
                            for item in value
                        ]
                    else:
                        config[field_name][key] = value
            else:
                config[field_name] = field_value

        # Add class information for deserialization
        config["_class"] = self.__class__.__name__
        config["_module"] = self.__class__.__module__

        return config

    @classmethod
    def from_config(cls: Type[T], config: Dict[str, Any]) -> T:
        # Remove class information
        config_copy = config.copy()
        config_copy.pop("_class", None)
        config_copy.pop("_module", None)

        # Restore numpy arrays from lists and dtypes
        for key, value in list(config_copy.items()):
            if isinstance(value, list):
                # Check if we have dtype information
                dtype_key = f"{key}_dtype"
                if dtype_key in config_copy:
                    dtype_str = config_copy.pop(dtype_key)
                    config_copy[key] = np.array(value, dtype=np.dtype(dtype_str))
                else:
                    # Check if list contains dictionaries that represent SerialisableDataClass objects
                    if value and isinstance(value[0], dict) and "_class" in value[0]:
                        # Reconstruct list of SerialisableDataClass objects
                        reconstructed_list = []
                        for item_config in value:
                            if (
                                isinstance(item_config, dict)
                                and "_class" in item_config
                            ):
                                # Import the class dynamically
                                module_name = item_config["_module"]
                                class_name = item_config["_class"]
                                module = __import__(module_name, fromlist=[class_name])
                                item_class = getattr(module, class_name)
                                reconstructed_list.append(
                                    item_class.from_config(item_config)
                                )
                            else:
                                reconstructed_list.append(item_config)
                        config_copy[key] = reconstructed_list
                    elif value and isinstance(value[0], tuple):
                        # Handle tuples that might contain SerialisableDataClass objects
                        reconstructed_list = []
                        for item_tuple in value:
                            if isinstance(item_tuple, (list, tuple)):
                                reconstructed_tuple = tuple(
                                    cls._reconstruct_if_serialisable(subitem)
                                    for subitem in item_tuple
                                )
                                reconstructed_list.append(reconstructed_tuple)
                            else:
                                reconstructed_list.append(item_tuple)
                        config_copy[key] = reconstructed_list
                    else:
                        config_copy[key] = np.array(value)
            elif isinstance(value, dict):
                # Check if dictionary represents a SerialisableDataClass object
                if "_class" in value:
                    # Import the class dynamically
                    module_name = value["_module"]
                    class_name = value["_class"]
                    module = __import__(module_name, fromlist=[class_name])
                    item_class = getattr(module, class_name)
                    config_copy[key] = item_class.from_config(value)
                else:
                    # Handle nested dictionaries
                    reconstructed_dict = {}
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict) and "_class" in subvalue:
                            # Import the class dynamically
                            module_name = subvalue["_module"]
                            class_name = subvalue["_class"]
                            module = __import__(module_name, fromlist=[class_name])
                            item_class = getattr(module, class_name)
                            reconstructed_dict[subkey] = item_class.from_config(
                                subvalue
                            )
                        elif isinstance(subvalue, list):
                            # Handle nested lists in dictionaries
                            reconstructed_list = []
                            for item in subvalue:
                                if isinstance(item, dict) and "_class" in item:
                                    module_name = item["_module"]
                                    class_name = item["_class"]
                                    module = __import__(
                                        module_name, fromlist=[class_name]
                                    )
                                    item_class = getattr(module, class_name)
                                    reconstructed_list.append(
                                        item_class.from_config(item)
                                    )
                                else:
                                    reconstructed_list.append(item)
                            reconstructed_dict[subkey] = reconstructed_list
                        else:
                            reconstructed_dict[subkey] = subvalue
                    config_copy[key] = reconstructed_dict
            elif key.endswith("_dtype"):
                # Remove dtype keys that weren't processed above
                if key in config_copy:
                    config_copy.pop(key)

        return cls(**config_copy)

    @staticmethod
    def _reconstruct_if_serialisable(value: Any) -> Any:
        if isinstance(value, dict) and "_class" in value:
            module_name = value["_module"]
            class_name = value["_class"]
            module = __import__(module_name, fromlist=[class_name])
            item_class = getattr(module, class_name)
            return item_class.from_config(value)
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_config(), indent=2)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        config = json.loads(json_str)
        return cls.from_config(config)
