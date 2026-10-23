"""The recorded leaf: a leaf that records another leaf's calls and replays them.

:class:`Recorded` satisfies :class:`~idia.core.Leaf` structurally, with no base class.
It files each call under the hash of the canonical wire request, the leaf's name and model,
the state and the questions in order, and keeps that key's calls in order, each a wire
response with its raw body or a provider error. A replay returns them in order and never
calls the wrapped leaf. A leaf's construction options are not in the key, so a test keeps
one log per leaf configuration and names it.

The modes:

- ``none`` replays only. A call with no recording, or past the number of calls recorded
  under its key, raises :class:`ReplayMiss`. The log must exist when the leaf is built.
- ``once`` records every call when the log does not exist yet, and otherwise behaves as
  ``none``.
- ``new`` replays a call that has a recording, records one that has none, and appends a
  call past its key's recorded count.

The log is one JSON file, rewritten whole after each recorded call. A replayed error is a
plain :class:`~idia.core.ProviderError` carrying the recorded fields. A replay runs
through the call loop like any call, so the attempts, outcomes and ``RequestFailed`` are
the loop's, made again; pass the dispatcher a zero-delay retry policy so a recorded retry
does not sleep again. The recorded leaf is not in the provider registry.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import tempfile
import typing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast, get_args

from idia.core import (
    Capabilities,
    Deadline,
    Leaf,
    ProviderError,
    TokenEstimator,
    WireRequest,
    WireResponse,
)
from idia.providers._canon import (
    CharacterEstimator,
    canonical_hash,
    decode,
    encode,
    plain,
    thaw,
)
from idia.providers.scripted import ScriptedCall

_Mode: TypeAlias = Literal["none", "once", "new"]
"""How the recorded leaf treats a call: replay only, record a fresh log, or add to one."""

_FORMAT: Final = 1
"""The log format this module reads and writes. A change to ``_canon``'s rendering, which
the log is written in, bumps it."""

_MODES: Final = frozenset(get_args(_Mode))

_Entry: TypeAlias = dict[str, object]
"""One recorded call as plain JSON: ``{"response": ...}`` or ``{"error": ...}``."""


class ReplayMiss(AssertionError):
    """A call the log holds no recording for, under a mode that may not record it.

    It is a test failure, not a :class:`~idia.core.ProviderError`: the call loop never
    retries it and it never becomes an outcome's error.
    """


class Recorded:
    """A leaf that records the calls of the leaf it wraps to ``log`` and replays them.

    ``name``, ``model``, ``capabilities`` and ``estimator`` are the wrapped leaf's. With no
    leaf, only mode ``none`` is allowed, ``capabilities`` is required, the name and model
    are read from the log and the estimator counts one token per four characters; a plan
    the leaf's own estimator shaped may then differ, so wrap the real leaf in mode ``none``
    whenever it matters. ``calls`` logs one :class:`~idia.providers.scripted.ScriptedCall`
    per call, in order, before it is answered.
    """

    name: str
    model: str
    capabilities: Capabilities
    estimator: TokenEstimator
    calls: list[ScriptedCall]

    def __init__(
        self,
        leaf: Leaf | None,
        log: Path,
        mode: _Mode,
        *,
        capabilities: Capabilities | None = None,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"{mode!r} is not a mode; use none, once or new")
        exists = log.exists()
        header = _load(log) if exists else None
        if leaf is None:
            if capabilities is None:
                raise TypeError("a recorded leaf with no leaf to wrap needs capabilities=")
            if mode != "none":
                raise ValueError(f"a recorded leaf with no leaf can only replay, not {mode!r}")
            if header is None:
                raise _missing(log)
            features = capabilities.enabled_features
            if header.features != features:
                raise ValueError(
                    f"{log} was recorded with features {sorted(header.features)}, "
                    f"not {sorted(features)}"
                )
            self.name, self.model = header.name, header.model
            self.capabilities = capabilities
            self.estimator = CharacterEstimator()
        else:
            if capabilities is not None:
                raise TypeError("capabilities= is the wrapped leaf's; pass it only with no leaf")
            if mode == "none" and header is None:
                raise _missing(log)
            self.name, self.model = leaf.name, leaf.model
            self.capabilities = leaf.capabilities
            self.estimator = leaf.estimator
            features = self.capabilities.enabled_features
            recorded = (header.name, header.model, header.features) if header else None
            if recorded is not None and recorded != (self.name, self.model, features):
                raise ValueError(
                    f"{log} was recorded from {recorded[0]}/{recorded[1]} with features "
                    f"{sorted(recorded[2])}, not {self.name}/{self.model} with {sorted(features)}"
                )
        self.calls = []
        self._leaf = leaf
        self._log = log
        self._recorded: dict[str, list[_Entry]] = header.calls if header is not None else {}
        self._fresh = mode == "once" and not exists
        self._records = self._fresh or mode == "new"
        self._seen: dict[str, int] = {}

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        """Log the call, then replay its recording or record it, as the mode says."""
        self.calls.append(ScriptedCall(wire_request=wire_request, deadline=deadline))
        key = canonical_hash(self.name, self.model, wire_request)
        index = self._seen.get(key, 0)
        self._seen[key] = index + 1
        entries = self._recorded.get(key, [])
        if not self._fresh and index < len(entries):
            return _replay(entries[index])
        if not self._records:
            raise ReplayMiss(
                f"{self._log} holds {len(entries)} calls for request {key}; "
                f"call {index + 1} has no recording"
            )
        return await self._record(key, wire_request, deadline)

    async def _record(
        self, key: str, wire_request: WireRequest, deadline: Deadline
    ) -> WireResponse:
        assert self._leaf is not None
        try:
            response = await self._leaf.call(wire_request, deadline)
        except ProviderError as error:
            self._store(key, _error(error))
            raise
        self._store(key, _response(response))
        return response

    def _store(self, key: str, entry: _Entry) -> None:
        """Append ``entry`` under ``key`` and rewrite the whole log through a temporary
        file, so a reader never sees half a log."""
        self._recorded.setdefault(key, []).append(entry)
        document = {
            "format": _FORMAT,
            "name": self.name,
            "model": self.model,
            "enabled_features": sorted(self.capabilities.enabled_features),
            "calls": self._recorded,
        }
        text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        self._log.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=self._log.parent, prefix=f".{self._log.name}.")
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
                file.write(text)
            os.replace(temporary, self._log)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Log:
    """A log as read: the configuration it was recorded under, the leaf's name, model and
    enabled features, and the recorded calls by key."""

    name: str
    model: str
    features: frozenset[str]
    calls: dict[str, list[_Entry]]


def _load(log: Path) -> _Log:
    """Read ``log``, refusing a format this module does not write."""
    document = json.loads(log.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{log} is not a recorded leaf's log")
    held = cast(dict[str, object], document)
    if held.get("format") != _FORMAT:
        raise ValueError(f"{log} is in format {held.get('format')!r}; this reads {_FORMAT}")
    return _Log(
        name=str(held["name"]),
        model=str(held["model"]),
        features=frozenset(cast(list[str], held["enabled_features"])),
        calls=cast(dict[str, list[_Entry]], held["calls"]),
    )


def _missing(log: Path) -> FileNotFoundError:
    return FileNotFoundError(f"mode none replays {log}, which does not exist")


def _error(error: ProviderError) -> _Entry:
    return {
        "error": {
            "message": error.message,
            "is_retryable": error.is_retryable,
            "retry_after": error.retry_after,
            "status_code": error.status_code,
            "request_id": error.request_id,
        }
    }


def _response(response: WireResponse) -> _Entry:
    """The wire response as JSON: through :func:`encode`, except the raw body and the raw
    usage, which are kept as the plain JSON they are."""
    usage = response.usage
    bare = replace(
        response,
        raw_body=None,
        usage=None if usage is None else replace(usage, raw_usage={}),
    )
    encoded = cast(dict[str, object], plain(encode(bare)))
    if usage is not None:
        cast(dict[str, object], encoded["usage"])["raw_usage"] = plain(usage.raw_usage)
    raw = response.raw_body
    if isinstance(raw, bytes):
        encoded["raw_body"] = {"base64": base64.b64encode(raw).decode()}
    elif raw is not None:
        encoded["raw_body"] = {"json": plain(raw)}
    return {"response": encoded}


def _replay(entry: _Entry) -> WireResponse:
    """The recorded wire response, or raise the recorded error as a plain ProviderError."""
    if "error" in entry:
        fields = cast(dict[str, typing.Any], entry["error"])
        raise ProviderError(
            fields["message"],
            is_retryable=fields["is_retryable"],
            retry_after=fields["retry_after"],
            status_code=fields["status_code"],
            request_id=fields["request_id"],
        )
    encoded = dict(cast(dict[str, object], entry["response"]))
    raw = cast(dict[str, object] | None, encoded.pop("raw_body", None))
    response = cast(WireResponse, decode({**encoded, "raw_body": None}, WireResponse))
    if raw is None:
        return response
    if "base64" in raw:
        return replace(response, raw_body=base64.b64decode(cast(str, raw["base64"])))
    return replace(response, raw_body=thaw(raw["json"]))
