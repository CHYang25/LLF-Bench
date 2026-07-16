import random
from llfbench.envs.metaworld.prompts import (
    close_gripper_feedback,
    open_gripper_feedback,
)
from llfbench.envs.metaworld.task_prompts.hp_feedback import hp_feedback as task_hp_feedback
from llfbench.envs.metaworld.task_prompts.hn_feedback import hn_feedback as task_hn_feedback
from llfbench.envs.metaworld.task_prompts.box_close_v2_prompts import (
    move_to_lid_feedback,
    lift_up_lid_feedback,
    move_to_box_feedback,
)
from llfbench.envs.metaworld.utils_prompts.conjunction_prompts import (
    positive_conjunctions_sampler,
    progressive_conjunction_sampler,
    negative_conjunctions_sampler
)
from llfbench.envs.metaworld.utils_prompts.direction_prompts import direction_desc_list
from llfbench.envs.metaworld.utils_prompts.recommend_prompts import recommend_templates_sampler
from llfbench.envs.metaworld.utils_prompts.degree_prompts import degree_adverbs

move_to_lid_and_lift_up_lid_prompts = (
    "You should first fetch the lid and lift it up.",
    "You should first get the lid and pick it up.",
    "You should first go to the lid and raise it.",
    "You should first fetch the lid and then lift it up.",
)

lift_up_lid_and_move_to_box_prompts = (
    "You should now lift up the lid and place it on the box.",
    "You should now pick up the lid and put it on the box.",
    "You should now raise the lid and close it on the box.",
    "You should now lift up the lid and then place it on the box.",
)

move_to_lid_and_move_to_box_prompts = (
    "You should first fetch the lid and place it on the box.",
    "You should first get the lid and put it on the box.",
    "You should first go to the lid and close it on the box.",
    "You should first fetch the lid and then place it on the box.",
)

move_to_lid_lift_up_lid_move_to_box_prompts = (
    "You should first fetch the lid, lift it up, and place it on the box.",
    "You should first get the lid, pick it up, and put it on the box.",
    "You should first go to the lid, raise it, and close it on the box.",
    "You should first fetch the lid, lift it up, and then place it on the box.",
    "You should first fetch the lid, lift it up, and put it on the box.",
    "You should first get the lid, pick it up, and place it on the box.",
    "You should first go to the lid, lift it up, and close it on the box.",
    "You should first fetch the lid, raise it, and place it on the box.",
)

perfect_feedback = (
    "the action is perfect for the state.",
    "the action is ideal for the state.",
    "the action is exactly right for the state.",
    "the action is best for the state.",
    "the action is optimal for the state.",
)

fair_feedback = (
    "the action is good but not perfect for the state.",
    "the action is helpful but not perfect for the state.",
    "the action is right but not perfect for the state.",
    "the action is useful but not perfect for the state.",
    "the action is working but not perfect for the state.",
)

neutral_feedback = (
    "the action is fair for the state.",
    "the action is acceptable for the state.",
    "the action is okay for the state.",
    "the action is reasonable for the state.",
    "the action is adequate for the state.",
)

flaw_feedback = (
    "the action is imperfect for the state.",
    "the action is not ideal for the state.",
    "the action is not quite right for the state.",
    "the action is not very helpful for the state.",
    "the action is not very useful for the state.",
)

poor_feedback = (
    "the action is poor for the state.",
    "the action is bad for the state.",
    "the action is wrong for the state.",
    "the action is not working for the state.",
    "the action is not good for the state.",
)

task_progress_dict = (
    move_to_lid_feedback,                          # 1 (001)
    lift_up_lid_feedback,                          # 2 (010)
    move_to_lid_and_lift_up_lid_prompts,           # 3 (011)
    move_to_box_feedback,                          # 4 (100)
    move_to_lid_and_move_to_box_prompts,           # 5 (101)
    lift_up_lid_and_move_to_box_prompts,           # 6 (110)
    move_to_lid_lift_up_lid_move_to_box_prompts,   # 7 (111)
)

action_optim_level = (
    perfect_feedback,
    fair_feedback,
    neutral_feedback,
    flaw_feedback,
    poor_feedback,
)

move_guidance_list = (
    direction_desc_list[0][0],
    direction_desc_list[0][1],
    direction_desc_list[1][0],
    direction_desc_list[1][1],
    direction_desc_list[2][0],
    direction_desc_list[2][1],
)

_TASK_PROGRESS_LOOKUP = {
    prompt[:-1]: 2 ** idx
    for idx, stage_prompts in enumerate((
        move_to_lid_feedback,
        lift_up_lid_feedback,
        move_to_box_feedback,
    ))
    for prompt in stage_prompts
}

_ACTION_OPTIM_LOOKUP = tuple(
    (prompt[:-1], idx)
    for idx, action_prompts in enumerate((
        task_hp_feedback,
        task_hn_feedback,
    ))
    for prompt in action_prompts
)

_GRIPPER_LOOKUP = tuple(
    (prompt[:-1].lower(), idx)
    for idx, gripper_prompts in enumerate((
        close_gripper_feedback,
        open_gripper_feedback,
    ))
    for prompt in gripper_prompts
)

_MOVE_GUIDANCE_LOOKUP = tuple(
    (prompt, idx)
    for idx, move_prompts in enumerate(move_guidance_list)
    for prompt in move_prompts
)

_DEGREE_ADVERBS = tuple(
    adverb
    for adverbs in degree_adverbs.values()
    for adverb in adverbs
)


class BoxCloseV2Merger:

    def __init__(self):
        pass

    def __call__(self, texts):
        if len(texts) == 1:
            return texts
        
        tp_score = 0
        ao_score = 0
        mg_set = []
        mg_seen = set()
        gripper_seen = False
        gripper_idx = None

        for text in texts:
            txt_list = text.split(".")
            if len(txt_list) == 2: # defensive: only action_optim present (carried over from parking)
                ao = txt_list[0]
                for action_prompt, action_idx in _ACTION_OPTIM_LOOKUP:
                    if action_prompt in ao:
                        ao_score += action_idx
                        break
                continue

            tp = txt_list[0]
            ao = txt_list[1]
            rest = txt_list[2:-1]

            tp_score |= _TASK_PROGRESS_LOOKUP[tp]

            for action_prompt, action_idx in _ACTION_OPTIM_LOOKUP:
                if action_prompt in ao:
                    ao_score += action_idx
                    break

            for _seg in rest:
                seg_lower = _seg.lower()
                matched_gripper = False
                for gripper_prompt, gripper_i in _GRIPPER_LOOKUP:
                    if gripper_prompt in seg_lower:
                        gripper_seen = True
                        gripper_idx = gripper_i  # last-seen wins
                        matched_gripper = True
                        break
                if matched_gripper:
                    continue

                for move_prompt, move_idx in _MOVE_GUIDANCE_LOOKUP:
                    if move_prompt in _seg:
                        if move_idx not in mg_seen:
                            mg_seen.add(move_idx)
                            mg_set.append(move_idx)
                        break

        task_progress = random.choice(task_progress_dict[tp_score-1])

        ao_score = ao_score / len(texts)
        level = min(int(ao_score * 5), 4)
        action_optim = random.choice(action_optim_level[level])

        gripper_text = ""
        if gripper_seen:
            gripper = random.choice(close_gripper_feedback if gripper_idx == 0 else open_gripper_feedback)
            gripper_text = negative_conjunctions_sampler() + gripper[0].lower() + gripper[1:]

        move_guidance = []
        for mg in mg_set:
            move_guidance.append(progressive_conjunction_sampler())
            move_guidance.append(recommend_templates_sampler().format(
                direction=random.choice(move_guidance_list[mg]),
                degree=random.choice(_DEGREE_ADVERBS),
            ))

        final_text = task_progress + positive_conjunctions_sampler() + action_optim + gripper_text + "".join(move_guidance)
        return final_text
