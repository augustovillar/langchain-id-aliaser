"""Dependency-free reversible ID aliasing engine.

The :class:`IdAliaser` detects long identifiers (UUIDs by default, plus any
patterns or explicit values you register), replaces them with short aliases such
as ``usr_a9Fk2`` before text reaches an LLM, and restores the originals on the
way back.

This module imports nothing from LangChain. It operates only on ``str``,
``dict`` and ``list`` structures, which is what lets the same engine serve
single model calls, agent middleware and raw graphs identically.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Pattern, Tuple

__all__ = ["IdAliaser", "UnknownAliasError"]

# Canonical UUID (any version), case-insensitive.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# base62 alphabet for compact, tokenizer-friendly aliases.
_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_DEFAULT_PREFIX = "id"
_MIN_HASH_LEN = 5


class UnknownAliasError(ValueError):
    """Raised by :meth:`IdAliaser.restore` in strict mode for an alias that
    looks generated but is not in the mapping (the model invented or mangled
    it)."""


def _base62(num: int) -> str:
    if num == 0:
        return _BASE62[0]
    out = []
    while num:
        num, rem = divmod(num, 62)
        out.append(_BASE62[rem])
    return "".join(reversed(out))


class IdAliaser:
    """Reversible aliaser for identifiers.

    Parameters
    ----------
    mode:
        ``"hash"`` (default) → deterministic alias derived from the value, so the
        same id always maps to the same alias with no stored state. ``"ordinal"``
        → sequential aliases (``id1``, ``id2`` …); shorter but order-dependent, so
        the mapping must be reused across turns to stay consistent.
    detect_uuids:
        Whether the built-in UUID pattern is active. ``True`` by default.
    default_prefix:
        Prefix used for aliases when no type is known. Defaults to ``"id"``.

    Notes
    -----
    An instance accumulates a bidirectional mapping as it sees values. For the
    stateless guarantee in ``"hash"`` mode, reuse one instance (or share its
    mapping via :meth:`export_mapping` / :meth:`load_mapping`) across a session.
    """

    def __init__(
        self,
        mode: str = "hash",
        detect_uuids: bool = True,
        default_prefix: str = _DEFAULT_PREFIX,
    ) -> None:
        if mode not in ("hash", "ordinal"):
            raise ValueError(f"mode must be 'hash' or 'ordinal', got {mode!r}")
        self.mode = mode
        self.default_prefix = default_prefix
        self._alias_to_original: Dict[str, str] = {}
        self._original_to_alias: Dict[str, str] = {}
        # value -> type prefix, for explicitly registered values.
        self._value_types: Dict[str, str] = {}
        self._patterns: List[Tuple[Pattern[str], str]] = []
        if detect_uuids:
            self._patterns.append((_UUID_RE, default_prefix))
        self._ordinal_counter = 0

    # -- registration ----------------------------------------------------

    def register(
        self,
        *,
        pattern: Optional[str] = None,
        value: Optional[str] = None,
        type: Optional[str] = None,
    ) -> None:
        """Register an extra id format.

        Provide either a ``pattern`` (regex matching id-shaped strings) or an
        explicit ``value``. ``type`` becomes the alias prefix (e.g. ``"usr"`` →
        ``usr_a9Fk2``); falls back to the default prefix when omitted.
        """
        if (pattern is None) == (value is None):
            raise ValueError("register requires exactly one of pattern= or value=")
        prefix = type or self.default_prefix
        if pattern is not None:
            self._patterns.append((re.compile(pattern), prefix))
        else:
            self._value_types[value] = prefix

    # -- mapping persistence --------------------------------------------

    def export_mapping(self) -> Dict[str, str]:
        """Return a copy of the ``{alias: original}`` mapping (e.g. to thread
        through graph state in ordinal mode)."""
        return dict(self._alias_to_original)

    def load_mapping(self, mapping: Dict[str, str]) -> None:
        """Merge a previously exported ``{alias: original}`` mapping back in."""
        for alias, original in mapping.items():
            self._alias_to_original[alias] = original
            self._original_to_alias[original] = alias
            self._ordinal_counter = max(
                self._ordinal_counter, self._ordinal_of(alias)
            )

    # -- alias generation -----------------------------------------------

    def _ordinal_of(self, alias: str) -> int:
        m = re.search(r"(\d+)$", alias)
        return int(m.group(1)) if m and self.mode == "ordinal" else 0

    def _make_alias(self, original: str, prefix: str) -> str:
        existing = self._original_to_alias.get(original)
        if existing is not None:
            return existing

        if self.mode == "ordinal":
            self._ordinal_counter += 1
            alias = f"{prefix}{self._ordinal_counter}"
        else:
            digest = hashlib.sha256(original.encode("utf-8")).digest()
            num = int.from_bytes(digest, "big")
            code = _base62(num)
            length = _MIN_HASH_LEN
            alias = f"{prefix}_{code[:length]}"
            # Resolve collisions deterministically by lengthening the hash.
            while (
                alias in self._alias_to_original
                and self._alias_to_original[alias] != original
            ):
                length += 1
                if length > len(code):
                    code += "0"
                alias = f"{prefix}_{code[:length]}"

        self._alias_to_original[alias] = original
        self._original_to_alias[original] = alias
        return alias

    def _prefix_for(self, value: str, pattern_prefix: str) -> str:
        return self._value_types.get(value, pattern_prefix)

    # -- core transform --------------------------------------------------

    def aliasify(self, obj: Any) -> Any:
        """Return a copy of ``obj`` with every detected id replaced by its alias.

        Recursively handles ``str``, ``dict`` and ``list``; other types pass
        through unchanged. Updates the internal mapping as a side effect.
        """
        if isinstance(obj, str):
            return self._aliasify_str(obj)
        if isinstance(obj, dict):
            return {k: self.aliasify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.aliasify(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.aliasify(v) for v in obj)
        return obj

    def _aliasify_str(self, text: str) -> str:
        # Explicit registered values first (they may not match any pattern).
        for value, prefix in self._value_types.items():
            if value and value in text:
                alias = self._make_alias(value, prefix)
                text = text.replace(value, alias)
        # Then pattern-based detection.
        for pattern, prefix in self._patterns:
            def _sub(match: "re.Match[str]") -> str:
                original = match.group(0)
                return self._make_alias(original, self._prefix_for(original, prefix))

            text = pattern.sub(_sub, text)
        return text

    def restore(self, obj: Any, strict: bool = False) -> Any:
        """Inverse of :meth:`aliasify`: replace known aliases with their original
        ids.

        Unknown aliases pass through untouched (forgiving). When ``strict`` is
        ``True``, an alias-shaped token that is not in the mapping raises
        :class:`UnknownAliasError`.
        """
        if isinstance(obj, str):
            return self._restore_str(obj, strict)
        if isinstance(obj, dict):
            return {k: self.restore(v, strict) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.restore(v, strict) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.restore(v, strict) for v in obj)
        return obj

    def _restore_str(self, text: str, strict: bool) -> str:
        if self._alias_to_original:
            # Replace longest aliases first to avoid partial-overlap clobbering.
            for alias in sorted(self._alias_to_original, key=len, reverse=True):
                if alias in text:
                    text = text.replace(alias, self._alias_to_original[alias])
        if strict:
            self._assert_no_unknown_aliases(text)
        return text

    def _alias_shape(self) -> Pattern[str]:
        # Matches tokens shaped like our generated aliases.
        prefixes = {self.default_prefix, *self._value_types.values()}
        prefixes.update(p for _, p in self._patterns)
        alt = "|".join(sorted(map(re.escape, prefixes), key=len, reverse=True))
        if self.mode == "ordinal":
            return re.compile(rf"\b(?:{alt})\d+\b")
        return re.compile(rf"\b(?:{alt})_[0-9A-Za-z]+\b")

    def _assert_no_unknown_aliases(self, text: str) -> None:
        for m in self._alias_shape().finditer(text):
            token = m.group(0)
            if token not in self._alias_to_original:
                raise UnknownAliasError(
                    f"Alias-shaped token {token!r} is not in the mapping; "
                    "the model may have invented or mangled it."
                )
