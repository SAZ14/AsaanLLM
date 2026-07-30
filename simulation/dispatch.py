"""Generic task dispatch: build dataclass-typed function arguments from a raw
JSON body, and serialize dataclass results back to JSON. This is what lets
service.py expose all 51 loan + ATM task types without 51 hand-written
Pydantic request models — the dataclasses in loan-agent / agents.atm already
carry the shape.
"""
from __future__ import annotations

import dataclasses
import inspect
import typing


def _resolve_hints(fn) -> dict:
    target = fn.__func__ if hasattr(fn, "__func__") else fn
    try:
        return typing.get_type_hints(target)
    except Exception:
        return {}


def dataclass_from_dict(cls, data):
    """Recursively build a dataclass instance from a plain dict, handling
    nested dataclasses and list[SomeDataclass] fields."""
    if not (dataclasses.is_dataclass(cls) and isinstance(cls, type)) or not isinstance(data, dict):
        return data
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        ftype = hints.get(f.name)
        origin = typing.get_origin(ftype) if ftype else None
        if origin in (list, typing.List) and isinstance(val, list):
            (inner,) = typing.get_args(ftype)
            val = [dataclass_from_dict(inner, v) if isinstance(v, dict) else v for v in val]
        elif ftype is not None and dataclasses.is_dataclass(ftype) and isinstance(val, dict):
            val = dataclass_from_dict(ftype, val)
        kwargs[f.name] = val
    return cls(**kwargs)


class MissingField(Exception):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"missing required field: {name}")


def build_args(fn, raw: dict) -> dict:
    """Build **kwargs for fn from a raw JSON dict, converting any dataclass-
    or list[dataclass]-typed parameters found in it. Scalars pass through."""
    hints = _resolve_hints(fn)
    sig = inspect.signature(fn)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name not in raw:
            if param.default is inspect.Parameter.empty:
                raise MissingField(name)
            continue
        val = raw[name]
        ftype = hints.get(name)
        origin = typing.get_origin(ftype) if ftype else None
        if origin in (list, typing.List) and isinstance(val, list):
            (inner,) = typing.get_args(ftype)
            val = [dataclass_from_dict(inner, v) if isinstance(v, dict) else v for v in val]
        elif ftype is not None and dataclasses.is_dataclass(ftype) and isinstance(val, dict):
            val = dataclass_from_dict(ftype, val)
        kwargs[name] = val
    return kwargs


def to_jsonable(obj):
    """Recursively turn dataclasses/tuples/lists/dicts into plain JSON-safe
    structures."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def call_render(render_fn, compute_kwargs: dict, facts):
    """Call a render_<task>_facts function whose arity varies: most take
    just (facts); a handful take (original_input, facts) where
    original_input is the sole compute-function argument. Optional trailing
    params (e.g. reject_bin_notes=None) are left at their defaults."""
    sig = inspect.signature(render_fn)
    required = [p for p in sig.parameters.values() if p.default is inspect.Parameter.empty]
    if len(required) <= 1:
        return render_fn(facts)
    if len(compute_kwargs) == 1:
        return render_fn(next(iter(compute_kwargs.values())), facts)
    first_name = required[0].name
    if first_name in compute_kwargs:
        return render_fn(compute_kwargs[first_name], facts)
    return render_fn(facts)


def _describe_type(ftype, depth: int = 0, max_depth: int = 3):
    """Recursively describe a type for the /tasks discovery endpoint: a
    dataclass becomes {type, fields: {name: <recursive>}}, list[dataclass]
    becomes {type, item_fields: {...}}, anything else is just its name."""
    if ftype is None:
        return "any"
    origin = typing.get_origin(ftype)
    if origin in (list, typing.List):
        args = typing.get_args(ftype)
        if not args:
            return "list"
        inner = args[0]
        if dataclasses.is_dataclass(inner) and isinstance(inner, type) and depth < max_depth:
            return {"type": f"list[{inner.__name__}]", "item_fields": _dataclass_fields_shape(inner, depth + 1, max_depth)}
        return f"list[{getattr(inner, '__name__', str(inner))}]"
    if dataclasses.is_dataclass(ftype) and isinstance(ftype, type) and depth < max_depth:
        return {"type": ftype.__name__, "fields": _dataclass_fields_shape(ftype, depth + 1, max_depth)}
    return getattr(ftype, "__name__", str(ftype))


def _dataclass_fields_shape(cls, depth: int, max_depth: int) -> dict:
    hints = typing.get_type_hints(cls)
    return {f.name: _describe_type(hints.get(f.name), depth, max_depth) for f in dataclasses.fields(cls)}


def describe_params(fn) -> dict:
    """Best-effort human-readable shape of fn's parameters, for a /tasks
    discovery endpoint. Not a full JSON schema — just enough to know what
    fields to send, recursing into nested dataclasses up to 3 levels deep."""
    hints = _resolve_hints(fn)
    sig = inspect.signature(fn)
    shape = {}
    for name, param in sig.parameters.items():
        if name in ("self", "use_llm"):
            continue
        ftype = hints.get(name)
        required = param.default is inspect.Parameter.empty
        desc = _describe_type(ftype, 0, 3)
        if isinstance(desc, dict):
            shape[name] = {**desc, "required": required}
        else:
            shape[name] = {"type": desc, "required": required}
    return shape


