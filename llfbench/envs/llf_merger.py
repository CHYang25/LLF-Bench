"""Generic base for multistep language-feedback mergers.

An env wrapper emits one structured *step record* per step and renders its own single-step
feedback text from that record.  A merger takes the records of an action chunk (H steps),
aggregates them into a window summary and renders one label from the same prompt pools.
Single-step text is therefore exactly the H=1 case of the merged text, so the two grammars
can never drift apart, and the merger never has to re-parse rendered prose.

Concrete mergers (see ``llfbench.envs.kitchen.multistep_merger``) implement three hooks:

* :meth:`BaseMultistepMerger.aggregate` -- records -> summary
* :meth:`BaseMultistepMerger.render`    -- summary -> text, choosing every word through ``fmt``
  so an integer paraphrase method is deterministic end to end
* :meth:`BaseMultistepMerger.parse`     -- single-step text -> record, the inverse of ``render``
  for a 1-record window; used when a dataset stored only text

``__call__`` is what the LLM-BC translators invoke on a ``(steps,)`` list of labels.  Its
items may be records, their ``to_dict`` dicts, their JSON strings, or plain text; it always
returns a ``str`` and never raises on unrecognised input (the discriminator feeds it freshly
generated free text), falling back to a space join of the texts.
"""
import functools
import json
from typing import Callable, Generic, List, Optional, Protocol, Sequence, Type, TypeVar, Union

from llfbench.envs.utils import format as format_prompt


class StepRecord(Protocol):
    def to_dict(self) -> dict: ...

    def to_json(self) -> str: ...

    @classmethod
    def from_dict(cls, d: dict) -> "StepRecord": ...


R = TypeVar("R")
S = TypeVar("S")


class BaseMultistepMerger(Generic[R, S]):
    #: The record dataclass this merger aggregates; set by the subclass.
    record_cls: Type[R] = None

    def __init__(self, paraphrase_method: Union[str, int] = 'random'):
        self.paraphrase_method = paraphrase_method
        #: ``fmt(pool, **slots) -> str``; same signature as ``LLFWrapper.format``.
        self._fmt = functools.partial(format_prompt, method=paraphrase_method)

    # ------------------------------------------------------------------ subclass hooks
    def parse(self, text: str) -> Optional[R]:
        """Single-step text -> record, or ``None`` when the text is not in the grammar."""
        return None

    def aggregate(self, records: List[R]) -> S:
        raise NotImplementedError

    def render(self, summary: S, fmt: Callable[..., str]) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------ fixed behaviour
    def coerce(self, item) -> Optional[R]:
        """Record / dict / JSON string / grammar text -> record; anything else -> ``None``."""
        try:
            if isinstance(item, self.record_cls):
                return item
            if isinstance(item, dict):
                return self.record_cls.from_dict(item)
            if isinstance(item, str):
                text = item.strip()
                if text.startswith('{'):
                    return self.record_cls.from_dict(json.loads(text))
                return self.parse(text)
        except Exception:
            return None
        return None

    def render_records(self, records: Sequence[R], fmt: Optional[Callable[..., str]] = None) -> str:
        """Aggregate + render.  ``fmt`` defaults to this merger's own paraphrase method."""
        return self.render(self.aggregate(list(records)), fmt or self._fmt)

    def __call__(self, items) -> str:
        if isinstance(items, str):
            return items
        items = list(items)
        if not items:
            return ''
        if len(items) == 1 and isinstance(items[0], str) and not items[0].lstrip().startswith('{'):
            # A lone piece of text is already a final label (e.g. a generated description).
            return items[0]
        records = [self.coerce(x) for x in items]
        if any(r is None for r in records):
            return ' '.join(x for x in items if isinstance(x, str))
        return self.render_records(records)
