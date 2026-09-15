from __future__ import annotations
from typing import TypeVar, Type, Callable, Generic

T = TypeVar("T")


class BaseRegistry(Generic[T]):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Each subclass gets its own registry dictionary
        cls._registry = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:

        def decorator(component_class: Type[T]) -> Type[T]:
            cls._registry[name] = component_class
            return component_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[T]:
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown {cls.__name__} '{name}'. Available: {available}")
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> T:
        component_class = cls.get(name)
        instance = component_class(*args, **kwargs)

        # Set registry key if the instance has the attribute
        if hasattr(instance, "_registry_key"):
            instance._registry_key = name

        return instance

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._registry.keys())
