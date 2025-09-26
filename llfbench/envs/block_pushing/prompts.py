
bp_instruction = (
    "Your job is to control a robot arm to solve a {task} task. You will get observations of the robot state and the world state in the form of json strings. Your objective is to provide control inputs to the robot to achieve the task's goal state over multiple time steps. Your actions are 2-dim vectors, which control the movement of the robot's end effector in the x and y directions. You action at each step sets the robot's target pose for that step in relative coordinate.",
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
    "You should go {expert_action}.",
    "I recommend that you move {expert_action}.",
    "You should move {expert_action}.",
    "I suggest you move {expert_action} relative to your current position.",
    "Moving {expert_action} will help you get to the goal faster.",
    "Try moving {expert_action}.",
    "One thing to try is to go {expert_action}.",
    "The direction {expert_action} looks promising.",
    "Aim to move {expert_action} at the next step.",
    "My advice is to move {expert_action}.",
    "Go {expert_action} relative to where you are now.",
    "I would try moving {expert_action} if I were you.",
    "Consider going {expert_action}.",
    "Attempt to move {expert_action} next.",
    "My suggestion is that you move {expert_action}.",
    "Moving {expert_action} next looks promising.",
    "I advise you to go {expert_action}.",
    "Next, move {expert_action}.",
    "Moving {expert_action} now is a good idea.",
    "If you want a tip, {expert_action} is a good direction to aim for next.",
    "I urge you to move {expert_action}.",
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
