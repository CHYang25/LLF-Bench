kt_instruction = (
    "You are controlling a Franka Panda arm in the {task} environment. "
    "Complete every kitchen subtask listed in the goal, such as opening the microwave, "
    "moving the kettle onto the burner, flipping the light switch or sliding the cabinet open. "
    "An action is a 7 dimensional vector in [-1, 1]: the first three entries are the change "
    "in the gripper's x, y and z world position (scaled by 0.2 metres), the next three are the "
    "change in its orientation as euler angles (scaled by 0.5 radians), and the last entry "
    "commands the gripper, where +1 opens it and -1 closes it.",
    "Your task is {task}. You move a Franka Panda arm around a kitchen and have to bring every "
    "object named in the goal to its target configuration. "
    "Each action is 7 numbers, all between -1 and 1. Entries 1-3 move the gripper along the "
    "world x, y and z axes by up to 0.2 metres, entries 4-6 rotate it by up to 0.5 radians "
    "about those axes, and entry 7 is the gripper command: +1 to open, -1 to close.",
    "This is the {task} task. Drive the Franka Panda arm so that all of the goal's kitchen "
    "subtasks are finished. "
    "Actions are 7 dimensional and clipped to [-1, 1]. The first three numbers are a "
    "displacement of the end effector in world coordinates, scaled by 0.2 metres; the next "
    "three rotate the end effector, scaled by 0.5 radians; the last number opens the gripper "
    "when it is positive and closes it when it is negative.",
)

r_feedback = (
    "Your reward for the latest step is {reward}.",
    "You got a reward of {reward}.",
    "The latest step brought you {reward} reward units.",
    "You've received a reward of {reward}.",
    "You've earned a reward of {reward}.",
    "You just got {reward} points.",
    "{reward} points for you.",
    "You've got yourself {reward} units of reward.",
    "The reward your latest step earned you is {reward}.",
    "The previous step's reward was {reward}.",
    "+{reward} reward",
    "Your reward is {reward}.",
    "The reward you just earned is {reward}.",
    "You have received {reward} points of reward.",
    "Your reward={reward}.",
    "The reward is {reward}.",
    "Alright, you just earned {reward} reward units.",
    "Your instantaneous reward is {reward}.",
    "Your rew. is {reward}.",
    "+{reward} points",
    "Your reward gain is {reward}."
)

fp_feedback = (
    "You should do the action {expert_action}.",
)

hn_feedback = (
    "the action is not good for the state.",
    "the action is not right for the state.",
    "the action is not helpful for the state.",
    "the action is not useful for the state.",
    "the action is not working for the state.",
    "the action is bad for the state.",
    "the action is wrong for the state.",
)

hp_feedback = (
    "the action is helpful for the state.",
    "the action is good for the state.",
    "the action is right for the state.",
    "the action is useful for the state.",
    "the action is working for the state.",
    "the action is correct for the state.",
    "the action is good for the state.",
)
