import random
from llfbench.envs.highway.prompts import (
    hp_feedback,
    hn_feedback
)
from llfbench.envs.highway.task_prompts.parking_prompts import (
    align_car_prompts,
    reach_goal_prompts,
    careful_park_prompts,
)
from llfbench.envs.highway.utils_prompts.conjunction_prompts import (
    positive_conjunctions_sampler, 
    progressive_conjunction_sampler,
    negative_conjunctions_sampler
)
from llfbench.envs.highway.utils_prompts.recommend_prompts import (
    accel_forward_recommend,
    accel_backward_recommend,
    deccel_recommend,
    turn_left_recommend,
    turn_right_recommend,
)

align_car_and_reach_gaol_prompts = (
    "You should first align your car to the parking spot and move to it.",
    "You should first line up your car to the parking spot and move to it.",
    "You should first align your car to the parking spot and go to it.",
    "You should first line up your car to the parking spot and go to it.",
)

reach_goal_and_careful_park_prompts = (
    "You should now move to the parking spot and carfully park your car into the spot.",
    "You should now go to the parking spot and carfully park your car into the spot.",
    "You should now move to the parking spot and slowly park your car into the spot.",
    "You should now go to the parking spot and slowly park your car into the spot.",
)

align_car_careful_park_prompts = (
    "You should first align your car and carefully park into the spot.",
    "You should first line up your car and carefully park into the spot.",
    "You should first align your car and slowly park into the spot.",
    "You should first line up your car and slowly park into the spot.",
)

align_car_reach_goal_careful_park_prompts = (
    "You should first aligh your car, move to the spot, and carefully park into the spot.",
    "You should first line up your car, go to the spot, and carefully park into the spot.",
    "You should first aligh your car, move to the spot, and carefully park into the spot.",
    "You should first line up your car, go to the spot, and carefully park into the spot.",
    "You should first aligh your car, move to the spot, and slowly park into the spot.",
    "You should first line up your car, go to the spot, and slowly park into the spot.",
    "You should first aligh your car, move to the spot, and slowly park into the spot.",
    "You should first line up your car, go to the spot, and slowly park into the spot.",
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
    align_car_prompts,
    reach_goal_prompts,
    align_car_and_reach_gaol_prompts,
    careful_park_prompts,
    align_car_careful_park_prompts,
    reach_goal_and_careful_park_prompts,
    align_car_reach_goal_careful_park_prompts,
)

action_optim_level = (
    perfect_feedback,
    fair_feedback,
    neutral_feedback,
    flaw_feedback,
    poor_feedback,
)

move_guidance_list = (
    accel_forward_recommend,
    accel_backward_recommend,
    deccel_recommend,
    turn_left_recommend,
    turn_right_recommend,
)

_TASK_PROGRESS_LOOKUP = {
    prompt[:-1]: 2 ** idx
    for idx, stage_prompts in enumerate((
        align_car_prompts,
        reach_goal_prompts,
        careful_park_prompts,
    ))
    for prompt in stage_prompts
}

_ACTION_OPTIM_LOOKUP = tuple(
    (prompt[:-1], idx)
    for idx, action_prompts in enumerate((
        hp_feedback,
        hn_feedback,
    ))
    for prompt in action_prompts
)

_MOVE_GUIDANCE_LOOKUP = tuple(
    (prompt[:-1], idx)
    for idx, move_prompts in enumerate(move_guidance_list)
    for prompt in move_prompts
)


class ParkingV0Merger:

    def __init__(self):
        pass

    def __call__(self, texts):
        tp_score = 0
        ao_score = 0
        mg_set = []
        mg_seen = set()

        for text in texts:
            txt_list = text.split(".")
            if len(txt_list) == 2: # if crashed, feedback would only have action_optim saying negative
                ao = txt_list[0]
                for action_prompt, action_idx in _ACTION_OPTIM_LOOKUP:
                    if action_prompt in ao:
                        ao_score += action_idx
                        break
                continue

            tp = txt_list[0]
            ao = txt_list[1]
            mg = txt_list[2:-1]

            tp_score |= _TASK_PROGRESS_LOOKUP[tp]

            for action_prompt, action_idx in _ACTION_OPTIM_LOOKUP:
                if action_prompt in ao:
                    ao_score += action_idx
                    break

            for _mg in mg:
                for move_prompt, move_idx in _MOVE_GUIDANCE_LOOKUP:
                    if move_prompt in _mg:
                        if move_idx not in mg_seen:
                            mg_seen.add(move_idx)
                            mg_set.append(move_idx)
                        break

        task_progress = random.choice(task_progress_dict[tp_score-1])
        
        ao_score = ao_score / len(texts)
        level = min(int(ao_score * 5), 4)
        action_optim = random.choice(action_optim_level[level])

        move_guidance = []
        for mg in mg_set:
            move_guidance.append(progressive_conjunction_sampler())
            move_guidance.append(random.choice(move_guidance_list[mg]))

        final_text = task_progress + positive_conjunctions_sampler() + action_optim + "".join(move_guidance)
        return final_text
