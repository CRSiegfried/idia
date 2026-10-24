"""The recorded leaf: a leaf that records another leaf's calls and replays them.

:class:`Recorded` satisfies :class:`~idia.core.Leaf` structurally, with no base class. It
files each call under the hash of the canonical wire request, the leaf's name and model, the
state, the questions in order and the request's options when there are any, and keeps that
key's calls in order, each a wire response with its raw body or a provider error. A replay
returns them in order and never calls the wrapped leaf. A leaf's construction options are
not in the key, so a test keeps one log per leaf configuration and names it.

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
import math
import os
import secrets
import stat
import typing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast, get_args

from idia.core import (
    DETAILS,
    WARNING_NO_SNAPSHOT,
    Capabilities,
    Deadline,
    Leaf,
    ProvenanceDetail,
    ProviderError,
    TokenEstimator,
    UnreportedReason,
    WireRequest,
    WireResponse,
    _held,  # pyright: ignore[reportPrivateUsage]
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

__all__ = [
    "Recorded",
    "ReplayMiss",
    "upgrade_log",
]

_Mode: TypeAlias = Literal["none", "once", "new"]
"""How the recorded leaf treats a call: replay only, record a fresh log, or add to one."""

_FORMAT: Final = 2
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
        file, so a reader never sees half a log. An entry the write refuses is not kept."""
        entries = self._recorded.setdefault(key, [])
        entries.append(entry)
        document = {
            "format": _FORMAT,
            "name": self.name,
            "model": self.model,
            "enabled_features": sorted(self.capabilities.enabled_features),
            "calls": self._recorded,
        }
        try:
            _write(self._log, document)
        except BaseException:
            entries.pop()
            if not entries:
                del self._recorded[key]
            raise


def _write(log: Path, document: typing.Mapping[str, object]) -> None:
    """Write ``document`` to ``log`` whole, sorted and indented, in UTF-8 with LF line
    endings, through a temporary file replaced in one step, so a reader never sees half a
    log. A log that exists keeps its permission bits, set before anything is written, and
    one that is a link is written where it points; a link to nothing is refused with
    ``ValueError``, and a new log takes the mode the process's umask gives a new file."""
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if log.is_symlink() and not log.exists():
        raise ValueError(f"{log} is a link to nothing; a log is written only where it points")
    log = log.resolve()
    try:
        mode: int | None = stat.S_IMODE(log.stat().st_mode) & 0o777
    except FileNotFoundError:
        mode = None
    log.parent.mkdir(parents=True, exist_ok=True)
    temporary = log.parent / f".{log.name}.{secrets.token_hex(8)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    handle = os.open(temporary, flags, 0o666 if mode is None else 0o600)
    try:
        if mode is not None:
            os.chmod(temporary, mode)
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, log)
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
    document: dict[str, object]


def _load(log: Path) -> _Log:
    """Read ``log``, refusing with ``ValueError`` a format this module does not write and a
    document that is not a recorded leaf's log, each entry read as a replay would read it."""
    text = log.read_text(encoding="utf-8")
    try:
        document: object = json.loads(text, parse_constant=_no_constant)
    except RecursionError:
        raise ValueError(f"{log} is not a recorded leaf's log: it nests too deep") from None
    except _Constant as error:
        raise ValueError(f"{log} is not a recorded leaf's log: {error}") from None
    if not isinstance(document, dict):
        raise ValueError(f"{log} is not a recorded leaf's log")
    held = cast(dict[str, object], document)
    if held.get("format") != _FORMAT:
        raise ValueError(f"{log} is in format {held.get('format')!r}; this reads {_FORMAT}")
    try:
        loaded = _Log(
            name=_string(held["name"]),
            model=_string(held["model"]),
            features=frozenset(_string(f) for f in cast(list[object], held["enabled_features"])),
            calls=_calls(held["calls"]),
            document=held,
        )
    except (KeyError, TypeError, AttributeError, ValueError, RecursionError) as error:
        raise ValueError(f"{log} is not a recorded leaf's log: {error!r}") from error
    return loaded


class _Constant(ValueError):
    """A ``NaN`` or infinity in a log, which JSON does not allow and the writer refuses."""


def _no_constant(name: str) -> object:
    raise _Constant(f"it holds {name}, which is not a JSON number")


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{value!r} is not a string")
    return value


def _calls(value: object) -> dict[str, list[_Entry]]:
    """The recorded calls, each entry checked by reading it as a replay does."""
    if not isinstance(value, dict):
        raise TypeError("the calls are not a mapping")
    calls = cast(dict[object, object], value)
    for key, entries in calls.items():
        if not isinstance(key, str) or not isinstance(entries, list):
            raise TypeError(f"the calls under {key!r} are not a list")
        for entry in cast(list[object], entries):
            if not isinstance(entry, dict):
                raise TypeError(f"an entry under {key!r} is not a mapping")
            checked = cast(_Entry, entry)
            if ("error" in checked) == ("response" in checked):
                raise TypeError(f"an entry under {key!r} holds not exactly one of error, response")
            if "error" in checked:
                _recorded_error(checked)
            else:
                _replay(checked)
    return cast(dict[str, list[_Entry]], calls)


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


def _recorded_error(entry: _Entry) -> ProviderError:
    """The recorded error as a plain ``ProviderError``."""
    fields = cast(dict[str, typing.Any], entry["error"])
    retry_after = fields["retry_after"]
    checks = {
        "message": isinstance(fields["message"], str),
        "is_retryable": isinstance(fields["is_retryable"], bool),
        "retry_after": retry_after is None
        or (
            isinstance(retry_after, int | float)
            and not isinstance(retry_after, bool)
            and math.isfinite(retry_after)
            and retry_after >= 0
        ),
        "status_code": fields["status_code"] is None
        or (isinstance(fields["status_code"], int) and not isinstance(fields["status_code"], bool)),
        "request_id": fields["request_id"] is None or isinstance(fields["request_id"], str),
    }
    wrong = [name for name, held in checks.items() if not held]
    if wrong:
        raise TypeError(f"a recorded error's {wrong} have the wrong type")
    return ProviderError(
        fields["message"],
        is_retryable=fields["is_retryable"],
        retry_after=fields["retry_after"],
        status_code=fields["status_code"],
        request_id=fields["request_id"],
    )


def _replay(entry: _Entry) -> WireResponse:
    """The recorded wire response, or raise the recorded error as a plain ProviderError."""
    if "error" in entry:
        raise _recorded_error(entry)
    encoded = dict(cast(dict[str, object], entry["response"]))
    raw = cast(dict[str, object] | None, encoded.pop("raw_body", None))
    response = cast(WireResponse, decode({**encoded, "raw_body": None}, WireResponse))
    if raw is None:
        return response
    if "base64" in raw:
        return replace(response, raw_body=base64.b64decode(cast(str, raw["base64"]), validate=True))
    return replace(response, raw_body=thaw(raw["json"]))


# ----------------------------------------------------------------------------------------
# Upgrading a log written before the provenance profile
# ----------------------------------------------------------------------------------------

_WIRE: Final = frozenset(
    {"system-one", "openai-decisions", "openai-chat", "cloudflare-clef", "cloudflare-clairic"}
)
"""The wire leaves: no runtime of their own, and every detail their wire left out not
reported."""

_GLINER_RUNTIME: Final = ("python", "gliner2", "torch", "transformers")
_GLINER_CLIENT: Final = ("huggingface_hub",)
_ALWAYS: Final[frozenset[ProvenanceDetail]] = frozenset({"elicitation", "calibrated"})
_PROVENANCE_KEYS: Final = frozenset({"versions", "client_versions", "unreported", "claimed"})


def upgrade_log(
    log: Path, *, reasons: typing.Mapping[ProvenanceDetail, UnreportedReason] | None = None
) -> int:
    """Bring a log written before the provenance profile up to it, offline, and return how
    many responses changed; a second run changes none.

    Each recorded response gets a reason for every profile detail that holds no value, by
    the rules its leaf, the log's ``name``, states: for a wire leaf (System One, OpenAI's
    two, Cloudflare's two) and the scripted leaf, ``versions`` ``not-applicable``, the
    snapshot ``not-pinned`` beside its ``no-snapshot`` warning, and every other missing
    detail ``not-reported``, but the passes ``not-exposed`` where the raw usage holds a pass
    count the recording idia did not read; for ``gliner``, ``versions`` and
    ``client_versions`` from the versions its raw body recorded, the output tokens
    ``not-applicable``, the other missing details as a wire leaf's, and ``model_reported``
    and a snapshot claimed, since a log cannot show the snapshot was resolved rather than
    given. Any other leaf takes its reasons from ``reasons``, the reason for each detail it
    gives none of.

    Nothing recorded changes: not an answer, a value, a warning or the raw body, and a
    reason, claim or version a response already holds stays. Error entries are left as they
    are. The log is refused with ``ValueError``, and nothing written, when a leaf has no
    rules and no ``reasons`` are given, ``reasons`` are given for a leaf with rules, give
    ``not-sent`` or give a missing snapshot a reason other than ``not-pinned``, a ``gliner``
    log records no runtime version, a missing
    snapshot has no ``no-snapshot`` warning, a response records an empty request id, its
    ``elicitation`` or ``calibrated`` holds no value or has a reason, or a detail would be
    left with neither a value nor a reason. The log keeps its format and keys, and is
    rewritten only if a response changed.
    """
    held = _load(log)
    document = held.document
    changed = 0
    for entries in held.calls.values():
        for index, entry in enumerate(entries):
            if "response" not in entry:
                continue
            before = _replay(entry)
            after = _upgraded(before, held.name, reasons, log)
            if after == before:
                continue
            upgraded = _response(after)
            _expect_only_provenance_added(entry, upgraded, log)
            entries[index] = upgraded
            changed += 1
    if changed:
        document["calls"] = held.calls
        _write(log, document)
    return changed


def _upgraded(
    response: WireResponse,
    name: str,
    reasons: typing.Mapping[ProvenanceDetail, UnreportedReason] | None,
    log: Path,
) -> WireResponse:
    """``response`` with the reasons, versions and claims its leaf's rules give it."""
    versions = dict(response.versions)
    client_versions = dict(response.client_versions)
    claimed = set(response.claimed)
    rules: dict[ProvenanceDetail, UnreportedReason]
    if reasons is not None and (name in _WIRE or name in {"scripted", "gliner"}):
        raise ValueError(f"{log}: {name!r} has rules of its own, so it takes no reasons")
    if name in _WIRE or name == "scripted":
        rules = {"versions": "not-applicable"}
    elif name == "gliner":
        recorded = _recorded_versions(response.raw_body) or {}
        if not versions:
            versions = {k: v for k, v in recorded.items() if k in _GLINER_RUNTIME}
            if not versions:
                raise ValueError(f"{log}: a gliner response records no runtime version")
        if not client_versions:
            client_versions = {k: v for k, v in recorded.items() if k in _GLINER_CLIENT}
        rules = {"usage.output_tokens": "not-applicable"}
        claimed |= {"model_reported"} if response.model_reported is not None else set()
        claimed |= {"snapshot"} if response.snapshot is not None else set()
    elif reasons is not None:
        rules = dict(reasons)
        if "not-sent" in rules.values():
            raise ValueError(f"{log}: not-sent is the dispatcher's, never a recorded response's")
    else:
        raise ValueError(
            f"{log}: no upgrade rules for the leaf {name!r}; pass the reasons it gives"
        )
    if rules.get("snapshot", "not-pinned") != "not-pinned":
        raise ValueError(
            f"{log}: a missing snapshot's reason is not-pinned, not {rules['snapshot']}"
        )
    if response.request_id == "":
        raise ValueError(f"{log}: a response records an empty request id")
    warned = any(w.code == WARNING_NO_SNAPSHOT for w in response.warnings)
    if response.snapshot is None and not warned and "snapshot" not in response.unreported:
        raise ValueError(f"{log}: a response has no snapshot and no no-snapshot warning")
    filled = replace(response, versions=versions, client_versions=client_versions)
    values = _held(filled)
    unreported = dict(response.unreported)
    for detail in sorted(_ALWAYS):
        if not values[detail] or detail in unreported:
            raise ValueError(f"{log}: a response leaves out {detail}, which is always a value")
    for detail in sorted(DETAILS):
        if values[detail] or detail in unreported:
            continue
        if detail == "snapshot":
            unreported[detail] = "not-pinned"
        elif detail in rules:
            unreported[detail] = rules[detail]
        elif name in _WIRE or name in {"scripted", "gliner"}:
            # A pass count the wire reported but the recording idia did not read.
            passes = detail == "usage.passes" and _raw_passes(filled)
            unreported[detail] = "not-exposed" if passes else "not-reported"
        else:
            raise ValueError(f"{log}: the reasons given for {name!r} leave out {detail}")
    return replace(filled, unreported=unreported, claimed=frozenset(claimed - set(unreported)))


def _raw_passes(response: WireResponse) -> bool:
    """Whether the raw usage holds a pass count that ``Usage.passes`` was not read from."""
    usage = response.usage
    if usage is None:
        return False
    passes = usage.raw_usage.get("passes")
    return isinstance(passes, int) and not isinstance(passes, bool)


def _recorded_versions(raw_body: object) -> dict[str, str] | None:
    """The versions a gliner raw body recorded, string values only, ``idia`` left out."""
    if not isinstance(raw_body, typing.Mapping):
        return None
    versions = cast(typing.Mapping[str, object], raw_body).get("versions")
    if not isinstance(versions, typing.Mapping):
        return None
    held = cast(typing.Mapping[object, object], versions)
    return {
        str(package): version
        for package, version in held.items()
        if isinstance(version, str) and package != "idia"
    }


def _expect_only_provenance_added(before: _Entry, after: _Entry, log: Path) -> None:
    """Refuse an upgrade that would change anything the log recorded."""
    old = cast(dict[str, object], before["response"])
    new = cast(dict[str, object], after["response"])
    kept = {key: value for key, value in new.items() if key not in _PROVENANCE_KEYS}
    if kept != {key: value for key, value in old.items() if key not in _PROVENANCE_KEYS}:
        raise ValueError(f"{log}: upgrading a response would change what it recorded")
