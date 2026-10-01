"""Tests for the FrankaKitchen structured feedback record, renderer and multistep merger.

Run everything::

    python tests/kitchen_feedback_test.py

Run one group::

    python tests/kitchen_feedback_test.py --only merger
"""

if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)

import functools
import json
import re

import numpy as np

from llfbench.envs.utils import format as format_prompt
from llfbench.envs.kitchen.multistep_merger import (
    MOVE_DEADBAND,
    TURN_DEADBAND,
    KitchenMultistepMerger,
    KitchenStepRecord,
)
from llfbench.envs.kitchen.task_prompts import franka_kitchen_prompts as kt
from llfbench.envs.kitchen.utils_prompts.degree_prompts import (
    MOVE_BUCKET_REPRESENTATIVE,
    TURN_BUCKET_REPRESENTATIVE,
    move_degree_bucket,
    turn_degree_bucket,
)


def fmt_k(k):
    return functools.partial(format_prompt, method=k)


def rec(**kw):
    return KitchenStepRecord(**kw)


def window(n, **kw):
    """n records that are identical except for t; per-field lists index by step."""
    out = []
    for i in range(n):
        fields = {k: (v[i] if isinstance(v, list) and len(v) == n and k not in
                      ('move_residual', 'move_flag', 'turn_residual', 'turn_flag', 'completed_order')
                      else v) for k, v in kw.items()}
        out.append(KitchenStepRecord(t=i, **fields))
    return out


SUBTASKS = list(kt.GOAL_PHRASES)
PHASES = list(kt.PHASE_GERUND)


# ---------------------------------------------------------------------------------------
# grammar
# ---------------------------------------------------------------------------------------

def test_record_round_trip():
    r = rec(t=3, subtask='kettle', phase='align', n_completed=1, completed_order=['microwave'],
            agree=False, move_residual=[np.float64(0.1), -0.2, 0.0], move_flag=[np.bool_(True), False, False],
            turn_residual=[0.0, 0.3, 0.0], turn_flag=[False, True, False], gripper='close')
    d = r.to_dict()
    json.dumps(d)  # JSON-safe
    assert all(type(x) is float for x in d['move_residual']), d['move_residual']
    assert all(type(x) is bool for x in d['move_flag']), d['move_flag']
    back = KitchenStepRecord.from_dict(d)
    assert back == r, (back, r)
    assert KitchenStepRecord.from_json(r.to_json()) == r
    extra = dict(d, unknown_key=1)
    assert KitchenStepRecord.from_dict(extra) == r, "from_dict must tolerate extra keys"
    print("  record round trip ......................... ok")


def test_phase_names_match_scripted_policy():
    from llfbench.envs.kitchen import scripted_policy as sp
    for name in ('ORIENT_FORWARD', 'SELECT_SUBTASK', 'MOVE_TO_PRECONTACT', 'ALIGN', 'APPROACH',
                 'CONTACT_OR_GRASP', 'MANIPULATE', 'KETTLE_TRANSPORT', 'RECEDE', 'VERIFY', 'RETREAT'):
        assert getattr(sp, name) in kt.PHASE_GERUND, name
        assert getattr(sp, name) in kt.PHASE_PAST, name
    assert sp.IDLE == 'idle'
    for phase in kt.PHASE_GERUND:
        g, p = kt.PHASE_GERUND[phase][0], kt.PHASE_PAST[phase][0]
        assert ('{object}' in g) == ('{object}' in p), phase
        assert ('{manipulation}' in g) == ('{manipulation}' in p), phase
    print("  phase names match scripted_policy ......... ok")


def _synthetic_single_records():
    """Records covering every subtask x phase, both verdicts and every guidance item type."""
    out = []
    for si, subtask in enumerate(SUBTASKS):
        for pi, phase in enumerate(PHASES):
            agree = (si + pi) % 2 == 0
            r = rec(subtask=subtask, phase=phase, agree=agree)
            axis = (si + pi) % 3
            if pi % 2 == 0:
                mag = list(MOVE_BUCKET_REPRESENTATIVE.values())[(si + pi) % 3]
                r.move_residual[axis] = mag if si % 2 else -mag
                r.move_flag[axis] = True
            if pi % 3 == 0:
                mag = list(TURN_BUCKET_REPRESENTATIVE.values())[(si + 2 * pi) % 3]
                r.turn_residual[axis] = -mag if pi % 2 else mag
                r.turn_flag[axis] = True
            if pi % 4 == 0:
                r.gripper = 'open' if si % 2 else 'close'
            out.append(r)
    # no-subtask phases, post-completion retreat, idle, no-verdict
    out.append(rec(subtask=None, phase='orient_forward', agree=True,
                   move_residual=[0.0, 0.0, 0.03], move_flag=[False, False, True]))
    out.append(rec(subtask=None, phase='select_subtask', agree=False))
    out.append(rec(subtask=None, phase='retreat', n_completed=1, completed_order=['microwave'], agree=True))
    out.append(rec(subtask=None, phase='idle', agree=True))
    out.append(rec(subtask='kettle', phase='approach'))  # agree None -> no verdict sentence
    return out


def _same_semantics(a, b):
    return (a.subtask, a.phase, a.agree, a.move_flag, a.turn_flag, a.gripper) == \
           (b.subtask, b.phase, b.agree, b.move_flag, b.turn_flag, b.gripper) and \
        all(np.sign(x) == np.sign(y) for x, y in zip(a.move_residual, b.move_residual)) and \
        all(np.sign(x) == np.sign(y) for x, y in zip(a.turn_residual, b.turn_residual)) and \
        all(move_degree_bucket(x) == move_degree_bucket(y) for x, y, f in zip(a.move_residual, b.move_residual, a.move_flag) if f) and \
        all(turn_degree_bucket(x) == turn_degree_bucket(y) for x, y, f in zip(a.turn_residual, b.turn_residual, a.turn_flag) if f)


def test_parse_round_trip():
    merger = KitchenMultistepMerger()
    n = 0
    for r in _synthetic_single_records():
        for k in range(3):
            text = merger.render_records([r], fmt=fmt_k(k))
            assert text[0].isupper() and text.endswith('.'), text
            assert len(re.split(r'(?<=\.) (?=[A-Z])', text)) <= 3, text
            back = merger.parse(text)
            assert back is not None, f"unparsable: {text!r}"
            assert _same_semantics(r, back), (r, back, text)
            assert merger.render_records([back], fmt=fmt_k(k)) == text, text
            n += 1
    assert merger.parse("garbage in") is None
    assert merger.parse("") is None
    win = window(4, subtask='microwave', phase=['move_to_precontact'] * 2 + ['align'] * 2, agree=True)
    assert merger.parse(merger.render_records(win, fmt=fmt_k(0))) is None, "merged grammar must not parse"
    print(f"  parse round trip ({n} labels) ............. ok")


def test_lookup_tables_are_unambiguous():
    table = KitchenMultistepMerger._stage_lookup()
    assert len(table) > 500, len(table)
    KitchenMultistepMerger._guidance_lookup()  # asserts uniqueness internally
    print("  lookup tables unambiguous ................. ok")


# ---------------------------------------------------------------------------------------
# merger
# ---------------------------------------------------------------------------------------

def _agg(records):
    return KitchenMultistepMerger().aggregate(records)


def test_stage_kinds():
    same = window(8, subtask='microwave', phase='approach', agree=True)
    assert _agg(same).kind == 'single'
    adv = window(8, subtask='microwave', phase=['move_to_precontact'] * 5 + ['align'] * 3, agree=True)
    s = _agg(adv)
    assert s.kind == 'two_phase' and s.phase_first == 'move_to_precontact' and s.phase_last == 'align'
    text = KitchenMultistepMerger().render(s, fmt_k(0))
    assert 'then squared the gripper up with it.' in text, text
    sw = window(8, subtask=['slide cabinet'] * 3 + ['kettle'] * 5, phase=['recede'] * 3 + ['move_to_precontact'] * 5,
                n_completed=[0] * 3 + [1] * 5, agree=True)
    for r in sw[3:]:
        r.completed_order = ['slide cabinet']
    s = _agg(sw)
    assert s.kind == 'switch' and s.done == 'slide cabinet' and s.subtask == 'kettle', s
    assert KitchenMultistepMerger().render(s, fmt_k(0)).startswith('You finished the slide cabinet and started on the kettle')
    # completion but nothing selected yet
    un = window(4, subtask=['microwave', 'microwave', None, None], phase=['recede', 'recede', 'retreat', 'retreat'],
                n_completed=[0, 0, 1, 1], agree=True)
    for r in un[2:]:
        r.completed_order = ['microwave']
    s = _agg(un)
    assert s.kind == 'switch_unselected' and s.done == 'microwave', s
    # completed_order lagging (kettle): fall back to first.subtask
    lag = window(4, subtask=['kettle', 'kettle', None, None], phase=['recede'] * 2 + ['retreat'] * 2,
                 n_completed=[0, 0, 1, 1], agree=True)
    assert _agg(lag).done == 'kettle'
    # reselection without completion
    mv = window(4, subtask=['kettle', 'kettle', 'microwave', 'microwave'],
                phase=['align', 'retreat', 'move_to_precontact', 'move_to_precontact'], agree=True)
    s = _agg(mv)
    assert s.kind == 'moved_on' and s.done == 'kettle' and s.subtask == 'microwave', s
    # the subtask was banked before the window (expert still receding from it): a switch
    early = window(4, subtask=['light switch', 'light switch', 'slide cabinet', 'slide cabinet'],
                   phase=['recede', 'recede', 'move_to_precontact', 'move_to_precontact'], n_completed=1, agree=True)
    for r in early:
        r.completed_order = ['light switch']
    s = _agg(early)
    assert s.kind == 'switch' and s.done == 'light switch', s
    # started from an unselected state
    st = window(4, subtask=[None, None, 'kettle', 'kettle'],
                phase=['orient_forward', 'select_subtask', 'move_to_precontact', 'move_to_precontact'], agree=True)
    assert _agg(st).kind == 'started'
    assert KitchenMultistepMerger().render(_agg(st), fmt_k(0)).startswith('You started on the kettle, moving the gripper toward the kettle handle.')
    # no subtask at either end
    ns = window(3, subtask=None, phase='orient_forward', agree=True)
    assert _agg(ns).kind == 'no_subtask'
    # post-completion retreat
    dr = window(3, subtask=None, phase='retreat', n_completed=1, agree=True)
    for r in dr:
        r.completed_order = ['light switch']
    s = _agg(dr)
    assert s.kind == 'done_retreat' and s.done == 'light switch'
    assert KitchenMultistepMerger().render(s, fmt_k(0)) == 'You are done with the light switch, backing the gripper away. Every action was right.'
    # idle
    idl = window(2, subtask=None, phase='idle', n_completed=4, agree=True)
    assert _agg(idl).kind == 'idle'
    # a completion banked by the env while the expert keeps working the same subtask is progress,
    # not a switch (the kettle/microwave are banked mid-manipulation)
    bank = window(8, subtask='slide cabinet', phase=['approach'] * 3 + ['manipulate'] * 5,
                  n_completed=[0] * 5 + [1] * 3, agree=True)
    for r in bank[5:]:
        r.completed_order = ['slide cabinet']
    s = _agg(bank)
    assert s.kind == 'two_phase', s
    same = window(4, subtask='slide cabinet', phase='manipulate', n_completed=[0, 0, 1, 1], agree=True)
    assert _agg(same).kind == 'single'
    print("  stage kinds ............................... ok")


def test_verdict_levels():
    m = KitchenMultistepMerger()
    one = window(1, subtask='kettle', phase='align', agree=False)
    assert m.render(m.aggregate(one), fmt_k(0)).endswith('The action was wrong.')
    allg = window(8, subtask='kettle', phase='align', agree=True)
    assert 'Every action was right.' in m.render(m.aggregate(allg), fmt_k(0))
    q = window(8, subtask='kettle', phase='align', agree=[False, False] + [True] * 6)
    assert 'Most actions were right, but some went wrong.' in m.render(m.aggregate(q), fmt_k(0))
    half = window(8, subtask='kettle', phase='align', agree=[False] * 4 + [True] * 4)
    assert 'Most actions were right, but some went wrong.' in m.render(m.aggregate(half), fmt_k(0))
    most = window(8, subtask='kettle', phase='align', agree=[False] * 6 + [True] * 2)
    assert 'Most actions were wrong.' in m.render(m.aggregate(most), fmt_k(0))
    none = window(3, subtask='kettle', phase='align')
    text = m.render(m.aggregate(none), fmt_k(0))
    assert text == 'You are moving the kettle onto the top left burner, squaring the gripper up with the kettle handle.', text
    # a reset record (agree None) inside a window does not count
    mixed = window(3, subtask='kettle', phase='align', agree=[None, True, True])
    assert 'Every action was right.' in m.render(m.aggregate(mixed), fmt_k(0))
    print("  verdict levels ............................ ok")


def test_guidance_union_and_deadband():
    m = KitchenMultistepMerger()
    recs = window(4, subtask='kettle', phase='align', agree=True)
    recs[0].move_flag[0] = True; recs[0].move_residual[0] = -0.05
    recs[3].move_residual[0] = -0.01                      # still open at the window end (very_low)
    recs[1].move_flag[2] = True; recs[1].move_residual[2] = 0.1
    recs[3].move_residual[2] = 0.002                      # corrected by the end -> dropped
    recs[2].turn_flag[1] = True; recs[2].turn_residual[1] = 0.3
    recs[3].turn_residual[1] = 0.25                       # medium
    recs[0].gripper = 'close'; recs[1].gripper = 'open'   # last non-None wins
    s = m.aggregate(recs)
    assert s.guidance == [('gripper', 'open'), ('move', 0, False, 'very_low'), ('turn', 1, True, 'medium')], s.guidance
    text = m.render(s, fmt_k(0))
    assert text.endswith('Open the gripper, move to the left gently and roll to the right firmly.'), text
    # two clauses -> "A and B."
    s.guidance = s.guidance[1:]
    assert m.render(s, fmt_k(0)).endswith('Move to the left gently and roll to the right firmly.')
    # deadbands
    assert MOVE_DEADBAND == 5e-3 and TURN_DEADBAND == 2e-2
    # min_flag_steps: an axis flagged on a single step is dropped at 2, kept at 1 (the default union)
    m2 = KitchenMultistepMerger(min_flag_steps=2)
    recs2 = window(4, subtask='kettle', phase='align', agree=True)
    recs2[0].move_flag[0] = True
    recs2[2].move_flag[1] = True; recs2[3].move_flag[1] = True
    for r in recs2:
        r.move_residual = [-0.05, 0.05, 0.0]
    assert [g[1] for g in m.aggregate(recs2).guidance] == [0, 1]
    assert [g[1] for g in m2.aggregate(recs2).guidance] == [1]
    assert m2.aggregate(recs2[:1]).guidance == [('move', 0, False, 'low')], "H=1 is never filtered"
    print("  guidance union + deadband ................. ok")


def test_call_contract():
    m = KitchenMultistepMerger(paraphrase_method=0)
    r1 = rec(subtask='kettle', phase='align', agree=True)
    r2 = rec(subtask='kettle', phase='approach', agree=False, gripper='close')
    assert m("plain text") == "plain text"
    assert m([]) == ''
    assert m(["a generated label."]) == "a generated label."
    merged = m([r1.to_json(), r2.to_dict()])
    assert merged == ('You are moving the kettle onto the top left burner. You squared the gripper up with the kettle '
                      'handle, then closed in on it. Most actions were right, but some went wrong. Close the gripper.'), merged
    assert isinstance(m([r1.to_json(), r2.to_dict(), r1]), str)
    assert m(["not grammar", r1.to_json()]) == "not grammar " + r1.to_json()
    assert isinstance(m([r1.to_json()]), str) and m([r1.to_json()]) == m.render_records([r1])
    # a single-step label in the new grammar round-trips through text
    text = m.render_records([r2])
    assert m([text, text]) == m.render_records([r2, r2])
    for item in ([{"garbage": 1}], [None], [1.5, r1]):
        out = m(item)
        assert isinstance(out, str), (item, out)
    print("  __call__ contract ......................... ok")


# ---------------------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------------------

def _make(feedback_type=('hp', 'hn', 'fp'), **kw):
    import gymnasium as gym
    import llfbench  # noqa: F401  (registers the env)
    from llfbench.envs.llf_env import LLFWrapper
    env = gym.make('llf-kitchen-FrankaKitchen-v1', instruction_type='b',
                   feedback_type=feedback_type, seed=0, warning=False, **kw)
    inner = env
    while not isinstance(inner, LLFWrapper):
        inner = inner.env
    return env, inner


def _label(feedback):
    return ' '.join(re.split(r'(?<=[.!?]) (?=[A-Z])', feedback.strip())[:-1])  # drop fp


def test_env_records_and_text():
    env, inner = _make()
    inner.set_paraphrase_method(0)
    merger = KitchenMultistepMerger(paraphrase_method=0)
    obs, info = env.reset(seed=0)
    r0 = info['feedback_record']
    assert r0 is not None and r0['agree'] is None and r0['t'] == 0 and r0['n_completed'] == 0
    assert not any(r0['move_flag']) and r0['gripper'] is None
    rng = np.random.default_rng(0)
    seen_hn = False
    records = []
    for t in range(30):
        a = inner.expert_action
        if t % 4 == 3:
            a = rng.uniform(-1, 1, size=a.shape)
        obs, _, term, trunc, info = env.step(a)
        rec_d = info['feedback_record']
        assert rec_d is not None and rec_d['t'] == t + 1
        assert (obs['feedback'] is not None)
        label = _label(obs['feedback'])
        assert label == merger.render_records([KitchenStepRecord.from_dict(rec_d)]), (label,)
        assert '[scripted_policy]' not in obs['feedback']
        assert merger.parse(label) is not None, label
        seen_hn |= rec_d['agree'] is False
        records.append(rec_d)
        if term or trunc:
            break
    assert seen_hn, "random actions should have produced at least one hn step"
    merged = merger(records[-8:])
    assert isinstance(merged, str) and merged.endswith('.')
    env.close()
    print("  env records and text ...................... ok")


def test_env_debug_and_expert_off():
    env, inner = _make(debug=True)
    env.reset(seed=0)
    obs, *_ = env.step(inner.expert_action)
    assert '[scripted_policy]' in obs['feedback'], "debug=True must still append the snapshot"
    env.close()
    env, inner = _make(inference_expert_action=False, feedback_type=('r', 'hp', 'hn', 'fp'))
    _, info = env.reset(seed=0)
    assert info['feedback_record'] is None
    obs, _, _, _, info = env.step(np.zeros(env.action_space.shape))
    assert info['feedback_record'] is None
    fb = obs['feedback']
    assert 'action was' not in fb and 'You are' not in fb, fb  # only the reward sentence survives
    env.close()
    print("  env debug / expert off .................... ok")


# ---------------------------------------------------------------------------------------

GROUPS = {
    "grammar": [
        test_record_round_trip,
        test_phase_names_match_scripted_policy,
        test_parse_round_trip,
        test_lookup_tables_are_unambiguous,
    ],
    "merger": [
        test_stage_kinds,
        test_verdict_levels,
        test_guidance_union_and_deadband,
        test_call_contract,
    ],
    "env": [
        test_env_records_and_text,
        test_env_debug_and_expert_off,
    ],
}


def main(only=None):
    failures = []
    for group, tests in GROUPS.items():
        if only and group != only:
            continue
        print(f"\n[{group}]")
        for test in tests:
            try:
                test()
            except AssertionError as exc:
                failures.append((test.__name__, exc))
                print(f"  {test.__name__} .... FAILED: {exc}")
    print("\n" + ("ALL PASSED" if not failures else f"{len(failures)} FAILED"))
    for name, exc in failures:
        print(f"  - {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(GROUPS), default=None)
    sys.exit(main(parser.parse_args().only))
