"""Structured step record, window summary and multistep merger for FrankaKitchen-v1.

``KitchenWrapper`` builds one :class:`KitchenStepRecord` per step and renders its hp/hn text
with ``KitchenMultistepMerger.render_records([record], fmt=self.format)``.  LLM-BC merges
the records of an action chunk with the same object, so the multistep label is the same
grammar summarising H steps:

    [Progress]   one stage sentence -- subtask + phase, or the first-and-last phases of the
                 window, or the subtask switch that happened inside it
    [Optimality] one verdict sentence -- the action (H=1) or the share of good actions
    [Guidance]   one sentence listing every correction still open at the end of the window

This module deliberately imports neither the wrapper nor ``scripted_policy`` (7k lines,
mujoco), so LLM-BC can instantiate the merger without a simulator.
"""
import dataclasses
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from llfbench.envs.llf_merger import BaseMultistepMerger
from llfbench.envs.kitchen.prompts import (
    verdict_all_good,
    verdict_most_bad,
    verdict_single_bad,
    verdict_single_good,
    verdict_some_bad,
)
from llfbench.envs.kitchen.task_prompts import franka_kitchen_prompts as kt
from llfbench.envs.kitchen.utils_prompts.degree_prompts import (
    MOVE_BUCKET_REPRESENTATIVE,
    TURN_BUCKET_REPRESENTATIVE,
    degree_adverbs,
    move_degree_bucket,
    turn_degree_bucket,
)
from llfbench.envs.kitchen.utils_prompts.direction_prompts import (
    move_direction_desc_list,
    move_direction_pool,
    turn_direction_desc_list,
    turn_direction_pool,
)
from llfbench.envs.kitchen.utils_prompts.recommend_prompts import (
    close_gripper_guidance,
    move_guidance,
    open_gripper_guidance,
    turn_guidance,
)

#: Below this the remaining Cartesian error on an axis is not worth a clause (metres).
MOVE_DEADBAND = 5e-3
#: Below this the expert is not asking for a meaningful wrist rotation (radians).
TURN_DEADBAND = 2e-2

# Phase names, as spelled by ``scripted_policy``.  Duplicated here (and checked against the
# policy in tests/kitchen_feedback_test.py) so this module stays free of the simulator.
PHASE_IDLE = "idle"
PHASE_RETREAT = "retreat"
PHASE_SELECT_SUBTASK = "select_subtask"


def _takes_object(phase: str) -> bool:
    pool = kt.PHASE_GERUND.get(phase)
    return bool(pool) and "{object}" in pool[0]


# ---------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------

@dataclass
class KitchenStepRecord:
    """Everything the language feedback of one wrapper step is derived from.

    JSON-safe: ``to_dict`` holds only Python scalars/lists, so the record can be stored
    beside the text in a dataset and rebuilt with ``from_dict`` / ``from_json``.
    """
    t: int = 0
    #: Goal element the expert is working on (``diagnostics['selected_subtask']``); ``None``
    #: while orienting / choosing / retreating after a completion.
    subtask: Optional[str] = None
    #: ``diagnostics['controller_phase']``.
    phase: str = PHASE_SELECT_SUBTASK
    #: ``len(kitchen.episode_task_completions)`` -- subtasks banked by the env.
    n_completed: int = 0
    #: ``diagnostics['completed_order']`` -- names the subtasks finished so far, in order.
    completed_order: List[str] = field(default_factory=list)
    #: hp (True) / hn (False) verdict of the step's action; ``None`` when there was no action.
    agree: Optional[bool] = None
    #: Signed metres from the gripper to the expert's waypoint after the step.
    move_residual: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    #: Axis got measurably worse during the step (the "moving away" test).
    move_flag: List[bool] = field(default_factory=lambda: [False, False, False])
    #: Signed radians of wrist rotation the expert still wants.
    turn_residual: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    #: The agent's wrist command fell short of the expert's on that axis.
    turn_flag: List[bool] = field(default_factory=lambda: [False, False, False])
    #: ``'open'`` / ``'close'`` when the agent's gripper command disagreed with the expert.
    gripper: Optional[str] = None
    version: int = 1

    def to_dict(self) -> dict:
        return dict(
            t=int(self.t),
            subtask=None if self.subtask is None else str(self.subtask),
            phase=str(self.phase),
            n_completed=int(self.n_completed),
            completed_order=[str(x) for x in self.completed_order],
            agree=None if self.agree is None else bool(self.agree),
            move_residual=[float(x) for x in self.move_residual],
            move_flag=[bool(x) for x in self.move_flag],
            turn_residual=[float(x) for x in self.turn_residual],
            turn_flag=[bool(x) for x in self.turn_flag],
            gripper=None if self.gripper is None else str(self.gripper),
            version=int(self.version),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "KitchenStepRecord":
        known = {f.name for f in dataclasses.fields(cls)}
        rec = cls(**{k: v for k, v in d.items() if k in known})
        rec.completed_order = list(rec.completed_order or [])
        rec.move_residual = [float(x) for x in rec.move_residual]
        rec.move_flag = [bool(x) for x in rec.move_flag]
        rec.turn_residual = [float(x) for x in rec.turn_residual]
        rec.turn_flag = [bool(x) for x in rec.turn_flag]
        return rec

    @classmethod
    def from_json(cls, text: str) -> "KitchenStepRecord":
        return cls.from_dict(json.loads(text))


@dataclass
class KitchenWindowSummary:
    """What :meth:`KitchenMultistepMerger.aggregate` distils a window of records into."""
    #: One of: single, two_phase, switch, switch_unselected, moved_on, started, no_subtask,
    #: done_retreat, idle.
    kind: str
    subtask: Optional[str]
    phase_first: Optional[str]
    phase_last: str
    #: The subtask that was finished (switch / done_retreat) or left (moved_on).
    done: Optional[str]
    n_scored: int
    n_bad: int
    #: ``('gripper', 'open'|'close')`` or ``('move'|'turn', axis, positive, bucket)``.
    guidance: List[tuple]


# ---------------------------------------------------------------------------------------
# Merger
# ---------------------------------------------------------------------------------------

class KitchenMultistepMerger(BaseMultistepMerger[KitchenStepRecord, KitchenWindowSummary]):
    record_cls = KitchenStepRecord

    def __init__(self, paraphrase_method='random', min_flag_steps: int = 1):
        """
        :param min_flag_steps: a move/turn axis enters the guidance sentence when it was flagged
            on at least this many steps of the window (and is still off at the window's end).
            ``1`` is the plain union of the per-step flags; the single-step flag sits at the
            IK jitter median by design, so over long windows a higher value filters that noise.
        """
        super().__init__(paraphrase_method)
        self.min_flag_steps = max(1, int(min_flag_steps))

    # -------------------------------------------------------------------- aggregate
    def aggregate(self, records: List[KitchenStepRecord]) -> KitchenWindowSummary:
        if not records:
            raise ValueError("cannot aggregate an empty window")
        first, last = records[0], records[-1]

        scored = [r for r in records if r.agree is not None]
        n_bad = sum(1 for r in scored if r.agree is False)

        # A completion banked inside the window only counts as a switch when the expert has
        # actually left the subtask: the env banks the kettle/microwave while the expert is
        # still manipulating or receding, and that window is ordinary progress on one subtask.
        completed_in_window = last.n_completed > first.n_completed
        newly = list(last.completed_order)[first.n_completed:]
        done = None
        if last.phase == PHASE_IDLE:
            kind = "idle"
        elif last.subtask is None:
            if completed_in_window:
                done = newly[-1] if newly else first.subtask
                kind = "switch_unselected"
            elif last.phase == PHASE_RETREAT and last.completed_order:
                done = last.completed_order[-1]
                kind = "done_retreat"
            else:
                kind = "no_subtask"
        elif first.subtask is None:
            kind = "started"
        elif first.subtask != last.subtask:
            # "finished" when the env banked it (inside this window or earlier, e.g. while the
            # expert was still receding from it); a bare reselection is a stall/abandonment.
            done = first.subtask
            finished = completed_in_window or first.subtask in last.completed_order
            kind = "switch" if finished else "moved_on"
        elif first.phase != last.phase:
            kind = "two_phase"
        else:
            kind = "single"

        guidance = []
        gripper = next((r.gripper for r in reversed(records) if r.gripper), None)
        if gripper:
            guidance.append(("gripper", gripper))
        need = min(self.min_flag_steps, len(records))
        for axis in range(3):
            n_flag = sum(1 for r in records if r.move_flag[axis])
            if n_flag >= need and abs(last.move_residual[axis]) > MOVE_DEADBAND:
                v = last.move_residual[axis]
                guidance.append(("move", axis, v >= 0, move_degree_bucket(v)))
        for axis in range(3):
            n_flag = sum(1 for r in records if r.turn_flag[axis])
            if n_flag >= need and abs(last.turn_residual[axis]) > TURN_DEADBAND:
                v = last.turn_residual[axis]
                guidance.append(("turn", axis, v >= 0, turn_degree_bucket(v)))

        return KitchenWindowSummary(
            kind=kind, subtask=last.subtask, phase_first=first.phase, phase_last=last.phase,
            done=done, n_scored=len(scored), n_bad=n_bad, guidance=guidance)

    # -------------------------------------------------------------------- render
    def render(self, summary: KitchenWindowSummary, fmt: Callable[..., str]) -> str:
        parts = [self._stage_sentence(summary, fmt)]
        verdict = self._verdict_sentence(summary, fmt)
        if verdict:
            parts.append(verdict)
        guidance = self._guidance_sentence(summary, fmt)
        if guidance:
            parts.append(guidance)
        return " ".join(parts)

    @staticmethod
    def _clause(fmt, phase: str, subtask: Optional[str], past: bool, object_word: Optional[str] = None) -> str:
        pool = (kt.PHASE_PAST if past else kt.PHASE_GERUND).get(phase)
        if pool is None:
            pool = ("worked on the subtask",) if past else ("working on the subtask",)
        kwargs = {}
        if "{object}" in pool[0]:
            kwargs["object"] = object_word or fmt(kt.OBJECT_PHRASES.get(subtask, kt.unknown_object_phrase))
        if "{manipulation}" in pool[0]:
            manip = (kt.MANIPULATION_PAST if past else kt.MANIPULATION_GERUNDS).get(
                subtask, kt.unknown_manipulation_past if past else kt.unknown_manipulation_gerund)
            kwargs["manipulation"] = fmt(manip)
        return fmt(pool, **kwargs)

    def _stage_sentence(self, s: KitchenWindowSummary, fmt) -> str:
        if s.kind == "idle":
            return fmt(kt.stage_idle)

        def goal():
            return fmt(kt.GOAL_PHRASES.get(s.subtask, kt.unknown_goal_phrase))

        def noun(name):
            return fmt(kt.SUBTASK_NOUNS.get(name, kt.unknown_subtask_noun))

        def now():
            return self._clause(fmt, s.phase_last, s.subtask, past=False)

        if s.kind == "single":
            return fmt(kt.stage_single, goal=goal(), clause=now())
        if s.kind == "two_phase":
            past1 = self._clause(fmt, s.phase_first, s.subtask, past=True)
            same_object = _takes_object(s.phase_first) and _takes_object(s.phase_last)
            past2 = self._clause(fmt, s.phase_last, s.subtask, past=True,
                                 object_word="it" if same_object else None)
            return fmt(kt.stage_two_phase, goal=goal(), past1=past1, past2=past2)
        if s.kind == "switch":
            return fmt(kt.stage_switch, done=noun(s.done), next=noun(s.subtask), clause=now())
        if s.kind == "moved_on":
            return fmt(kt.stage_moved_on, done=noun(s.done), next=noun(s.subtask), clause=now())
        if s.kind == "switch_unselected":
            return fmt(kt.stage_switch_unselected, done=noun(s.done))
        if s.kind == "started":
            return fmt(kt.stage_started, next=noun(s.subtask), clause=now())
        if s.kind == "done_retreat":
            return fmt(kt.stage_done_retreat, done=noun(s.done), clause=now())
        if s.kind == "no_subtask":
            return fmt(kt.stage_no_subtask, clause=now())
        raise ValueError(f"unknown stage kind {s.kind!r}")

    @staticmethod
    def _verdict_sentence(s: KitchenWindowSummary, fmt) -> Optional[str]:
        if s.n_scored == 0:
            return None
        if s.n_scored == 1:
            return fmt(verdict_single_bad if s.n_bad else verdict_single_good)
        frac = s.n_bad / s.n_scored
        if frac == 0:
            return fmt(verdict_all_good)
        if frac <= 0.5:
            return fmt(verdict_some_bad)
        return fmt(verdict_most_bad)

    @staticmethod
    def _guidance_sentence(s: KitchenWindowSummary, fmt) -> Optional[str]:
        clauses = []
        for item in s.guidance:
            if item[0] == "gripper":
                clauses.append(fmt(open_gripper_guidance if item[1] == "open" else close_gripper_guidance))
            elif item[0] == "move":
                _, axis, positive, bucket = item
                clauses.append(fmt(move_guidance,
                                   direction=fmt(move_direction_pool(axis, positive)),
                                   degree=fmt(degree_adverbs[bucket])))
            elif item[0] == "turn":
                _, axis, positive, bucket = item
                clauses.append(fmt(turn_guidance,
                                   direction=fmt(turn_direction_pool(axis, positive)),
                                   degree=fmt(degree_adverbs[bucket])))
            else:
                raise ValueError(f"unknown guidance item {item!r}")
        if not clauses:
            return None
        body = clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + " and " + clauses[-1]
        return body[0].upper() + body[1:] + "."

    # -------------------------------------------------------------------- parse
    _stage_lookup_cache: Optional[Dict[str, Tuple[Optional[str], str, Optional[str]]]] = None
    _verdict_lookup_cache: Optional[Dict[str, bool]] = None
    _guidance_lookup_cache: Optional[dict] = None

    @classmethod
    def _stage_lookup(cls) -> Dict[str, Tuple[Optional[str], str, Optional[str]]]:
        """Every single-step stage sentence -> (subtask, phase, done)."""
        if cls._stage_lookup_cache is not None:
            return cls._stage_lookup_cache
        table: Dict[str, Tuple[Optional[str], str, Optional[str]]] = {}

        def put(sentence, value):
            prev = table.get(sentence)
            assert prev is None or prev == value, f"ambiguous stage sentence {sentence!r}: {prev} vs {value}"
            table[sentence] = value

        def clauses(phase, subtask):
            out = []
            for tpl in kt.PHASE_GERUND[phase]:
                objects = kt.OBJECT_PHRASES.get(subtask, kt.unknown_object_phrase) if "{object}" in tpl else (None,)
                manips = kt.MANIPULATION_GERUNDS.get(subtask, kt.unknown_manipulation_gerund) if "{manipulation}" in tpl else (None,)
                for o in objects:
                    for m in manips:
                        out.append(tpl.format(object=o, manipulation=m))
            return out

        for subtask, goals in kt.GOAL_PHRASES.items():
            for goal in goals:
                for phase in kt.PHASE_GERUND:
                    for clause in clauses(phase, subtask):
                        for frame in kt.stage_single:
                            put(frame.format(goal=goal, clause=clause), (subtask, phase, None))
        for phase, pool in kt.PHASE_GERUND.items():
            if "{object}" in pool[0] or "{manipulation}" in pool[0]:
                continue
            for clause in clauses(phase, None):
                for frame in kt.stage_no_subtask:
                    put(frame.format(clause=clause), (None, phase, None))
        for done, nouns in kt.SUBTASK_NOUNS.items():
            for noun in nouns:
                for clause in clauses(PHASE_RETREAT, None):
                    for frame in kt.stage_done_retreat:
                        put(frame.format(done=noun, clause=clause), (None, PHASE_RETREAT, done))
        for sentence in kt.stage_idle:
            put(sentence, (None, PHASE_IDLE, None))
        cls._stage_lookup_cache = table
        return table

    @classmethod
    def _verdict_lookup(cls) -> Dict[str, bool]:
        if cls._verdict_lookup_cache is None:
            table = {s: True for s in verdict_single_good}
            table.update({s: False for s in verdict_single_bad})
            cls._verdict_lookup_cache = table
        return cls._verdict_lookup_cache

    @classmethod
    def _guidance_lookup(cls) -> dict:
        if cls._guidance_lookup_cache is None:
            gripper = {s: "open" for s in open_gripper_guidance}
            gripper.update({s: "close" for s in close_gripper_guidance})
            move_dir, turn_dir = {}, {}
            for axis in range(3):
                for positive in (False, True):
                    for word in move_direction_desc_list[axis][int(positive)]:
                        assert word not in move_dir, f"duplicate move direction {word!r}"
                        move_dir[word] = (axis, positive)
                    for word in turn_direction_desc_list[axis][int(positive)]:
                        assert word not in turn_dir, f"duplicate turn direction {word!r}"
                        turn_dir[word] = (axis, positive)
            adverb = {}
            for bucket, words in degree_adverbs.items():
                for w in words:
                    assert w not in adverb, f"duplicate adverb {w!r}"
                    adverb[w] = bucket
            cls._guidance_lookup_cache = dict(gripper=gripper, move_dir=move_dir, turn_dir=turn_dir, adverb=adverb)
        return cls._guidance_lookup_cache

    def parse(self, text: str) -> Optional[KitchenStepRecord]:
        sentences = [s.strip() for s in re.split(r"(?<=\.) (?=[A-Z])", text.strip()) if s.strip()]
        if not sentences:
            return None
        stage = self._stage_lookup().get(sentences[0])
        if stage is None:
            return None
        subtask, phase, done = stage
        rec = KitchenStepRecord(subtask=subtask, phase=phase)
        if done is not None:
            rec.completed_order = [done]
            rec.n_completed = 1
        verdicts = self._verdict_lookup()
        for sentence in sentences[1:]:
            if sentence in verdicts:
                rec.agree = verdicts[sentence]
            elif not self._parse_guidance(sentence, rec):
                return None
        return rec

    def _parse_guidance(self, sentence: str, rec: KitchenStepRecord) -> bool:
        tables = self._guidance_lookup()
        body = sentence[:-1] if sentence.endswith(".") else sentence
        body = body[:1].lower() + body[1:]
        for clause in re.split(r", | and ", body):
            if clause in tables["gripper"]:
                rec.gripper = tables["gripper"][clause]
                continue
            m = re.match(r"^move (.+) (\w+)$", clause)
            if m and m.group(1) in tables["move_dir"] and m.group(2) in tables["adverb"]:
                axis, positive = tables["move_dir"][m.group(1)]
                bucket = tables["adverb"][m.group(2)]
                rec.move_residual[axis] = (1.0 if positive else -1.0) * MOVE_BUCKET_REPRESENTATIVE[bucket]
                rec.move_flag[axis] = True
                continue
            m = re.match(r"^(.+) (\w+)$", clause)
            if m and m.group(1) in tables["turn_dir"] and m.group(2) in tables["adverb"]:
                axis, positive = tables["turn_dir"][m.group(1)]
                bucket = tables["adverb"][m.group(2)]
                rec.turn_residual[axis] = (1.0 if positive else -1.0) * TURN_BUCKET_REPRESENTATIVE[bucket]
                rec.turn_flag[axis] = True
                continue
            return False
        return True
