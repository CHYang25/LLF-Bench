ad_instruction = (
    "Your job is to control a dexterous robot hand to manipulate an object and achieve the task goal over multiple time steps. You will get the robot hand state, including the angular positions of the wrist and finger joints, as well as object-related state information and goal-related state information, in the form of json strings. The hand state describes the configuration of the wrist and all fingers, including the forefinger, middle finger, ring finger, little finger, and thumb. Depending on the task, the object-related state may include the object's position, linear velocity, angular velocity, orientation, and its relative distance or rotation with respect to the goal. Your objective is to coordinate the wrist and finger joints to grasp, control, and manipulate the object so that it reaches the desired goal state. Your actions are continuous control vectors, which specify the actuator commands for the wrist and finger joints of the robot hand.",
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
