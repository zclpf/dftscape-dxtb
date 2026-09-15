from __future__ import annotations
from typing import cast

from ..common import BaseRegistry
from ..core.interfaces import Backend


class BackendRegistry(BaseRegistry):

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Backend:
        return cast(Backend, super().create(name, *args, **kwargs))
