pt_instruction = (
"""
Your job is to use a circular end-effector (blue) to push a T-shaped block (gray) to a fixed target (red) using point contacts in a 2-D environment setting.
Observations are provided in one of two formats: either RGB images with end-effector location, or nine 2D keypoints representing the T-shaped block along with the same proprioceptive data.
You will get observations provided in one of the predefined formats and in the form of json strings.
Your actions are a 2-dim vector, which are the desired target pose from the policy.
Specifically, you should executes low-level motion using a PD (Proportional-Derivative) controller. Given the desired target position from the policy (i.e., the action), the controller computes an acceleration based on the position error and the current velocity of the end-effector. The velocity is then updated using the computed acceleration and the timestep.
Your objective is to provide control signals to the circular end-effector to achieve the task's goal state over multiple time steps.
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

hp_feedback = (
    "the action is helpful based on the state because {reason}.",
    "this action aligns with the current state, as {reason}.",
    "the state supports this action because {reason}.",
    "this action is appropriate given the state because {reason}.",
    "the state indicates this action is beneficial since {reason}.",
    "the state confirms this action is effective since {reason}.",
    "the state makes this a suitable action because {reason}.",
    "the state shows this action is advantageous, as {reason}.",
    "this action is validated by the state, as {reason}.",
    "the action is effective based on the state because {reason}.",
    "this action fits well with the state conditions since {reason}.",
    "the state suggests this action is optimal, as {reason}.",
    "this action leverages the current state successfully, as {reason}.",
    "the state provides a basis for this action success, as {reason}.",
    "this action is justified by the state context, as {reason}.",
    "the state enables this action to perform well, because {reason}.",
    "this action corresponds to the state requirements, since {reason}.",
    "the state reinforces the value of this action, because {reason}.",
    "this action capitalizes on the state features since {reason}.",
    "the state renders this action favorable, as {reason}.",
    "this action is well-suited to the state dynamics, as {reason}.",
    "the state backs this action as a good choice because {reason}.",
    "this action thrives under the current state since {reason}.",
    "the state highlights the merit of this action because {reason}.",
    "this action is a strong match for the state since {reason}.",
    "this action is perfectly suited for the state, as {reason}.",
    "the state clearly favors this action, as {reason}.",
    "choosing this action is wise given the state because {reason}.",
    "this action maximizes the state potential since {reason}.",
    "the state and this action are in harmony because {reason}.",
    "this action is the best response to the state since {reason}.",
    "the state makes this action highly effective, as {reason}.",
    "this action is a perfect fit for the state, as {reason}."
)

hn_feedback = (
    "based on the state the action is not helping, since {reason}.",
    "the action is ineffective based on the state, as {reason}.",
    "the state shows this action is unhelpful, due to {reason}.",
    "the state indicates this action is not effective, because {reason}.",
    "this action does not benefit the current state, since {reason}.",
    "the state confirms this action is unsuitable because {reason}.",
    "the state makes this action ineffective, for {reason}.",
    "the state shows this action is disadvantageous, as {reason}.",
    "the state validates that this action is not helpful, due to {reason}.",
    "the state reveals this action is unproductive, because {reason}.",
    "this action falls short given the state conditions, since {reason}.",
    "the state suggests this action is suboptimal because {reason}.",
    "this action fails to utilize the current state, as {reason}.",
    "the state undermines the success of this action, as {reason}.",
    "this action is misaligned with the state context, due to {reason}.",
    "the state hinders this action performance, because {reason}.",
    "this action clashes with the state requirements, since {reason}.",
    "the state exposes the flaws in this action because {reason}.",
    "this action struggles under the current state, for {reason}.",
    "the state renders this action impractical, as {reason}.",
    "this action is poorly suited to the state dynamics, due to {reason}.",
    "the state contradicts the value of this action, because {reason}.",
    "this action is weakened by the state features, since {reason}.",
    "the state signals this action as a poor fit because {reason}.",
    "this action falters given the state constraints, for {reason}.",
    "this action does not match what the state requires, as {reason}.",
    "the state does not justify this action, due to {reason}.",
    "this action is a poor choice for the current state, because {reason}.",
    "the state and this action are at odds, since {reason}.",
    "this action fails to address the state needs because {reason}.",
    "choosing this action here is not advisable, for {reason}.",
    "this action is not the right move for this state, as {reason}.",
    "the state makes this action less effective, due to {reason}."
)