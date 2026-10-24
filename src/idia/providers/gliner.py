"""The GLiNER2 leaf: ``fastino/gliner2.5-small-v1`` run in this process, for ``extraction`` of
text. Registered as ``gliner``, installed by the ``gliner`` extra.

The runtime is imported here, at the top, so that importing this module without the extra
fails with ``ModuleNotFoundError`` and the registry's guard names the extra to install.
gliner2 imports torch only when a model class is first used, so torch is imported here
too. The leaf's logic lives in an unlisted module that imports neither.
"""

from __future__ import annotations

import gliner2  # pyright: ignore[reportMissingImports, reportUnusedImport] # noqa: F401
import torch  # pyright: ignore[reportMissingImports, reportUnusedImport] # noqa: F401

from idia.providers._gliner import Gliner, GlinerRuntime

__all__ = [
    "Gliner",
    "GlinerRuntime",
]
