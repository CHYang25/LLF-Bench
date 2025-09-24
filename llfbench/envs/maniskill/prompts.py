ms_instruction = (
"""
Your job is to control a Panda robot arm to solve a {task} task. 
You will get observations of the robot state and the world state in the form of json strings. 
Your objective is to provide control inputs to the robot to achieve the task's goal state over multiple time steps. 
Your actions are 8-dim vectors, where the first 7 dimensions control the angular movement of the robot's joints, 
and the last dimension controls the gripper state (0 means opening it, and 1 means closing it).
""",
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
    "You should go to {expert_action}.",
    "I recommend that you move to {expert_action}.",
    "You should move to {expert_action}.",
    "I suggest you move to pose {expert_action}.",
    "The pose {expert_action} will help you get to the goal faster.",
    "Try moving to pose {expert_action}.",
    "One thing to try is to go to {expert_action}.",
    "The target {expert_action} is promising.",
    "Aim to reach pose {expert_action} at the next step.",
    "My advice is to reach first {expert_action}.",
    "Go for the post {expert_action}.",
    "I would try the target {expert_action} if I were you.",
    "Consider going to {expert_action}.",
    "Attempt to reach pose {expert_action} next.",
    "My suggestion is that you go towards pose {expert_action}.",
    "Moving to pose {expert_action} next looks promising.",
    "I advise you to go to {expert_action}.",
    "Next, move to pose {expert_action}.",
    "Moving to {expert_action} now is a good idea.",
    "If you want a tip, {expert_action} is a good pose to aim for next.",
    "I urge you to move to pose {expert_action}.",
)

open_gripper_feedback = (
    "You should open the gripper.",
    "You need to open the gripper.",
    "You can open the gripper.",
)

close_gripper_feedback = (
    "You should close the gripper.",
    "You need to close the gripper.",
    "You can close the gripper.",
)

hp_feedback = (
    "the action is helpful for the state, since {reason}.",
    "the action is helpful for the state because {reason}.",
    "the action is helpful for the state, as {reason}.",
    "the action is good for the state, since {reason}.",
    "the action is good for the state because {reason}.",
    "the action is good for the state, as {reason}.",
    "the action is right for the state, since {reason}.",
    "the action is right for the state because {reason}.",
    "the action is right for the state, as {reason}.",
    "the action is useful for the state, since {reason}.",
    "the action is useful for the state because {reason}.",
    "the action is useful for the state, as {reason}.",
    "the action is working for the state, since {reason}.",
    "the action is working for the state because {reason}.",
    "the action is working for the state, as {reason}.",
    "the action is correct for the state, since {reason}.",
    "the action is correct for the state because {reason}.",
    "the action is correct for the state, as {reason}.",
)

hn_feedback = (
    "the action is not good for the state, since {reason}.",
    "the action is not good for the state because {reason}.",
    "the action is not good for the state, as {reason}.",
    "the action is not right for the state, since {reason}.",
    "the action is not right for the state because {reason}.",
    "the action is not right for the state, as {reason}.",
    "the action is not helpful for the state, since {reason}.",
    "the action is not helpful for the state because {reason}.",
    "the action is not helpful for the state, as {reason}.",
    "the action is not useful for the state, since {reason}.",
    "the action is not useful for the state because {reason}.",
    "the action is not useful for the state, as {reason}.",
    "the action is not working for the state, since {reason}.",
    "the action is not working for the state because {reason}.",
    "the action is not working for the state, as {reason}.",
    "the action is bad for the state, since {reason}.",
    "the action is bad for the state because {reason}.",
    "the action is bad for the state, as {reason}.",
    "the action is wrong for the state, since {reason}.",
    "the action is wrong for the state because {reason}.",
    "the action is wrong for the state, as {reason}.",
)
