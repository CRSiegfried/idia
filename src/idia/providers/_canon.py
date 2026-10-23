"""The test doubles' deterministic renderings: Python data to JSON-ready data and back.

:func:`encode` turns idia's shapes into JSON-ready data, the same output for the same
input on every run; :func:`decode` rebuilds them, guided by the type a field declares and
never by a class named in the data: a dataclass's type tag only picks among the classes the
declared type already admits, and only classes on a closed list are ever built.
:func:`canonical_hash` is the key the recorded leaf files a call under. The character
estimator both doubles default to lives here too.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime
import enum
import hashlib
import json
import math
import types
import typing
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, cast

from idia.core import (
    Box,
    BoxAnswer,
    ExtractionAnswer,
    Fault,
    ImagePart,
    JsonPart,
    JsonValue,
    Label,
    LabelAnswer,
    NoulAnswer,
    Question,
    Refusal,
    Request,
    RuleFailure,
    Span,
    State,
    TextPart,
    TokenEstimate,
    Usage,
    Warning,
    WireRequest,
    WireResponse,
)

_CHARS_PER_TOKEN: Final = 4

_NON_FINITE: Final = {
    repr(math.inf): math.inf,
    repr(-math.inf): -math.inf,
    repr(math.nan): math.nan,
}
"""The values a tagged non-finite float holds, and the floats they stand for."""

_TIMES: Final[tuple[tuple[str, type], ...]] = (
    ("datetime", datetime.datetime),
    ("date", datetime.date),
    ("time", datetime.time),
    ("timedelta", datetime.timedelta),
)
"""The date and time types :func:`encode` tags by name, a datetime before the date it is."""

_DECODABLE: Final[frozenset[object]] = frozenset(
    {
        LabelAnswer,
        NoulAnswer,
        ExtractionAnswer,
        BoxAnswer,
        Refusal,
        Fault,
        RuleFailure,
        Box,
        Span,
        Warning,
        Usage,
        WireResponse,
        WireRequest,
        State,
        TextPart,
        JsonPart,
        ImagePart,
        Question,
        Label,
    }
)
"""The only dataclasses :func:`decode` builds."""


class Json:
    """Stands in for ``JsonValue`` when a field's declared type is resolved: ``JsonValue``
    is a string alias at run time, so it cannot be resolved as written."""


# ----------------------------------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------------------------------


def json_bytes(value: JsonValue) -> bytes:
    """``value`` as sorted, compact JSON."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _left_out(value: object, name: str) -> bool:
    """Whether ``encode`` leaves a field out: a request's empty ``options``, a usage's
    ``passes`` when ``None``, so renderings made before either existed are unchanged."""
    if name == "options" and isinstance(value, (Request, WireRequest)):
        return not value.options
    return name == "passes" and isinstance(value, Usage) and value.passes is None


def encode(value: object) -> JsonValue:
    """Python data as JSON-ready data, the same output for the same input on every run.

    A dataclass becomes an object tagged with its type name; a date, time, datetime or
    timedelta an object tagged with its type's name, holding its ISO form or its seconds;
    a non-finite float an object tagged ``float`` holding its ``repr``, so no string is
    ever read as a float; bytes base64; an enum member its value; a set a sorted array.
    A request's empty ``options`` are left out, and a usage's ``passes`` when ``None``, so a
    rendering made before either existed is unchanged. Any other type raises
    :class:`TypeError`.

    The recorded leaf's logs and the HTTP harness's cassettes are written in this
    rendering: a change to it bumps the recorded leaf's ``_FORMAT`` and the cassettes'
    ``FORMAT``, and the golden tests in ``tests/test_canon.py`` with them.
    """
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, enum.Enum):
        return encode(value.value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"type": "float", "value": repr(value)}
    if isinstance(value, bytes):
        return base64.b64encode(value).decode()
    for tag, kind in _TIMES:
        if isinstance(value, kind):
            if isinstance(value, datetime.timedelta):
                return {"type": tag, "value": value.total_seconds()}
            return {"type": tag, "value": cast(datetime.date, value).isoformat()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        encoded = {
            f.name: encode(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if not _left_out(value, f.name)
        }
        return {"type": type(value).__name__, **encoded}
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): encode(item) for key, item in mapping.items()}
    if isinstance(value, frozenset | set):
        members = cast(Iterable[object], value)
        return tuple(sorted((encode(item) for item in members), key=json_bytes))
    if isinstance(value, tuple | list):
        items = cast(Iterable[object], value)
        return tuple(encode(item) for item in items)
    raise TypeError(f"the test doubles cannot render a {type(value).__name__}")


def canonical_hash(name: str, model: str, wire_request: WireRequest) -> str:
    """The key a recorded call is filed under: the SHA-256, in hex, of the leaf's name and
    model, the state, the questions in order and the options, rendered as sorted, compact
    JSON. Empty options are left out, so a request without options keeps the key it had
    before requests carried them."""
    held: dict[str, object] = {
        "name": name,
        "model": model,
        "state": wire_request.state,
        "questions": wire_request.questions,
    }
    if wire_request.options:
        held["options"] = wire_request.options
    canonical = encode(held)
    return hashlib.sha256(json_bytes(canonical)).hexdigest()


# ----------------------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------------------


def decode(value: object, type_hint: object) -> typing.Any:
    """Rebuild what :func:`encode` wrote for a value of ``type_hint``.

    The declared type decides what is built: a tagged object becomes the dataclass of that
    name only among those ``type_hint`` admits and only from the closed list; an array
    becomes a tuple or a frozenset; a string becomes a ``Rule`` or a non-finite float where
    the type asks for one. A part declared ``JsonValue`` is thawed as plain JSON, its arrays
    made tuples, and never read for tags. Anything the type does not admit raises
    :class:`TypeError`.
    """
    if type_hint is Json:
        return thaw(value)
    if type_hint is typing.Any or type_hint is object:
        return value
    origin = typing.get_origin(type_hint)
    args = typing.get_args(type_hint)
    if origin is typing.Union or origin is types.UnionType:
        return _union(value, args)
    if origin is typing.Literal:
        if value not in args:
            raise TypeError(f"{value!r} is not one of {args}")
        return value
    if origin is tuple:
        items = _array(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode(item, args[0]) for item in items)
        if len(items) != len(args):
            raise TypeError(f"{len(items)} items for a tuple of {len(args)}")
        return tuple(decode(item, arg) for item, arg in zip(items, args, strict=True))
    if origin is frozenset:
        return frozenset(decode(item, args[0]) for item in _array(value))
    if origin is Mapping or origin is dict:
        mapping = _object(value)
        return {key: decode(item, args[1]) for key, item in mapping.items()}
    if type_hint in _DECODABLE:
        return _dataclass(_object(value), cast(type, type_hint))
    if isinstance(type_hint, type):
        return _type(value, type_hint)
    raise TypeError(f"cannot decode a value declared {type_hint!r}")


def _type(value: object, kind: type) -> object:
    if kind is type(None):
        if value is not None:
            raise TypeError(f"{value!r} is not None")
        return None
    if issubclass(kind, enum.Enum):
        return kind(value)
    if kind is float:
        if isinstance(value, Mapping):
            tagged = cast(Mapping[str, object], value)
            held = tagged.get("value")
            if tagged.get("type") != "float" or held not in _NON_FINITE:
                raise TypeError(f"{value!r} is not a non-finite float")
            return _NON_FINITE[cast(str, held)]
        if isinstance(value, int | float) and not isinstance(value, bool):
            return value
        raise TypeError(f"{value!r} is not a float")
    if kind is bytes:
        if not isinstance(value, str):
            raise TypeError(f"{value!r} is not base64")
        return base64.b64decode(value, validate=True)
    for tag, time_kind in _TIMES:
        if kind is time_kind:
            return _time(_object(value), tag, time_kind)
    if kind in (bool, int, str):
        if type(value) is not kind and not (kind is int and type(value) is bool):
            raise TypeError(f"{value!r} is not a {kind.__name__}")
        return value
    raise TypeError(f"{kind.__name__} is not a type the test doubles rebuild")


def _time(value: Mapping[str, object], tag: str, kind: type) -> object:
    if value.get("type") != tag:
        raise TypeError(f"{value!r} is not a {tag}")
    held = value.get("value")
    if kind is datetime.timedelta:
        if not isinstance(held, int | float):
            raise TypeError(f"{held!r} is not a number of seconds")
        return datetime.timedelta(seconds=held)
    if not isinstance(held, str):
        raise TypeError(f"{held!r} is not an ISO form")
    if kind is datetime.datetime:
        return datetime.datetime.fromisoformat(held)
    if kind is datetime.date:
        return datetime.date.fromisoformat(held)
    return datetime.time.fromisoformat(held)


def _dataclass(value: Mapping[str, object], kind: type) -> object:
    if value.get("type") != kind.__name__:
        raise TypeError(f"{value.get('type')!r} is not a {kind.__name__}")
    hints = typing.get_type_hints(kind, localns={"JsonValue": Json})
    unknown = set(value) - {"type"} - {field.name for field in dataclasses.fields(kind)}
    if unknown:
        raise TypeError(f"{kind.__name__} does not admit {sorted(unknown)}")
    held = {
        field.name: decode(value[field.name], hints[field.name])
        for field in dataclasses.fields(kind)
        if field.name in value
    }
    return kind(**held)


def _union(value: object, args: Sequence[object]) -> object:
    """Pick the member of the union the value was written as, then decode it as that."""
    if value is None:
        if type(None) in args:
            return None
        raise TypeError("None is not admitted here")
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        tag = mapping.get("type")
        for arg in args:
            if isinstance(arg, type) and arg.__name__ == tag and arg is not type(None):
                return decode(mapping, arg)
        for arg in args:
            if arg is Json or typing.get_origin(arg) in (Mapping, dict):
                return decode(mapping, arg)
        raise TypeError(f"no member of {args} is tagged {tag!r}")
    if isinstance(value, list | tuple):
        items = cast(Sequence[object], value)
        for arg in args:
            if arg is Json or typing.get_origin(arg) in (tuple, frozenset):
                return decode(items, arg)
        raise TypeError(f"no member of {args} is an array")
    if isinstance(value, str):
        for arg in args:
            if arg is str or arg is Json:
                return value
            if isinstance(arg, type) and issubclass(arg, enum.Enum):
                return arg(value)
            if typing.get_origin(arg) is typing.Literal and value in typing.get_args(arg):
                return value
        if bytes in args:
            return decode(value, bytes)
        raise TypeError(f"no member of {args} is a string")
    for kind in (bool, int, float):
        if type(value) is kind:
            if kind in args or Json in args:
                return value
            if kind is int and float in args:
                return value
    raise TypeError(f"{value!r} fits no member of {args}")


def _array(value: object) -> Sequence[object]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{value!r} is not an array")
    return cast(Sequence[object], value)


def _object(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{value!r} is not an object")
    return cast(Mapping[str, object], value)


def thaw(value: object) -> JsonValue:
    """Plain JSON as a ``JsonValue``: its arrays made tuples, nothing else changed."""
    if isinstance(value, list | tuple):
        return tuple(thaw(item) for item in cast(Sequence[object], value))
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return {key: thaw(item) for key, item in mapping.items()}
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"{type(value).__name__} is not JSON")


def plain(value: JsonValue) -> object:
    """A ``JsonValue`` as plain JSON for :mod:`json`: tuples made lists, views made dicts."""
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    return value


# ----------------------------------------------------------------------------------------
# The character estimator
# ----------------------------------------------------------------------------------------


def tokens(text: str) -> int:
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def render_wire_request(wire_request: WireRequest) -> str:
    """The wire request as sorted, compact JSON: the prompt the scripted leaf "sent"."""
    return json_bytes(encode(wire_request)).decode()


class CharacterEstimator:
    """The doubles' default estimator: one token per four characters of the rendered wire
    request, with no margin."""

    def estimate(self, state: State, questions: Sequence[Question]) -> TokenEstimate:
        rendered = render_wire_request(WireRequest(state=state, questions=tuple(questions)))
        return TokenEstimate(tokens=tokens(rendered), margin=0.0)
