
# This file contains the prompts for the verbal instructions and feedback.

b_instruction = (
    "Find the best action as fast as possible. Your action is an integer between {low} and {high}.",
    "Quickly identify the optimal action. Your action should be an integer within the range of {low} and {high}.",
    "Determine the most suitable action swiftly. Your action must be an integer from {low} to {high}.",
    "Locate the prime action as rapidly as you can. Your action is an integer that falls between {low} and {high}.",
    "Discover the top action in the shortest time. Your action is an integer that lies between {low} and {high}.",
    "Uncover the superior action with speed. Your action is an integer within the boundaries of {low} and {high}.",
    "Ascertain the finest action promptly. Your action is an integer between the values of {low} and {high}.",
    "Pinpoint the preeminent action without delay. Your action is an integer that ranges from {low} to {high}.",
    "Spot the paramount action in no time. Your action is an integer between the limits of {low} and {high}.",
    "Identify the supreme action at a fast pace. Your action is an integer from the range of {low} to {high}.",
    "Detect the optimal action as quickly as you can. Your action is an integer between the range of {low} and {high}.",
    "Find the most effective action swiftly. Your action should be an integer between {low} and {high}.",
    "Seek the best action rapidly. Your action must be an integer from {low} to {high}.",
    "Search for the ideal action as fast as possible. Your action is an integer that falls between {low} and {high}.",
    "Look for the perfect action in the shortest time. Your action is an integer that lies between {low} and {high}.",
    "Find the most suitable action quickly. Your action is an integer within the boundaries of {low} and {high}.",
    "Determine the best action as soon as possible. Your action is an integer between the values of {low} and {high}.",
    "Locate the optimal action without delay. Your action is an integer that ranges from {low} to {high}.",
    "Discover the prime action in no time. Your action is an integer between the limits of {low} and {high}.",
    "Uncover the top action at a fast pace. Your action is an integer from the range of {low} to {high}.",
    "Ascertain the superior action promptly. Your action is an integer between the range of {low} and {high}.",
)

p_instruction = (
    "Hint: Action {bad_action} is not the right one, as it gets an expected reward of {reward}.",
    "Suggestion: The action {bad_action} is incorrect as it receives an expected reward of {reward}.",
    "Note: The action {bad_action} is not appropriate since it obtains an expected reward of {reward}.",
    "Advice: The action {bad_action} is not suitable as it yields an expected reward of {reward}.",
    "Tip: The action {bad_action} is not correct as it acquires an expected reward of {reward}.",
    "Pointer: The action {bad_action} is not accurate as it gains an expected reward of {reward}.",
    "Indication: The action {bad_action} is not precise as it secures an expected reward of {reward}.",
    "Guidance: The action {bad_action} is not exact as it earns an expected reward of {reward}.",
    "Direction: The action {bad_action} is not proper as it achieves an expected reward of {reward}.",
    "Instruction: The action {bad_action} is not fitting as it attains an expected reward of {reward}.",
    "Information: The action {bad_action} is not apt as it procures an expected reward of {reward}.",
    "Clue: The action {bad_action} is not valid as it gets an expected reward of {reward}.",
    "Insight: The action {bad_action} is not acceptable as it receives an expected reward of {reward}.",
    "Observation: The action {bad_action} is not right as it obtains an expected reward of {reward}.",
    "Recommendation: The action {bad_action} is not ideal as it yields an expected reward of {reward}.",
    "Suggestion: The action {bad_action} is not perfect as it acquires an expected reward of {reward}.",
    "Hint: The action {bad_action} is not suitable as it gains an expected reward of {reward}.",
    "Advice: The action {bad_action} is not accurate as it secures an expected reward of {reward}.",
    "Tip: The action {bad_action} is not exact as it earns an expected reward of {reward}.",
    "Pointer: The action {bad_action} is not proper as it achieves an expected reward of {reward}.",
    "Indication: The action {bad_action} is not fitting as it attains an expected reward of {reward}.",
)

c_instruction = (
    "Hint: The optimal action is {best_arm}",
    "Suggestion: The best action to take is {best_arm}",
    "Tip: The most suitable action is {best_arm}",
    "Advice: The ideal action is {best_arm}",
    "Note: The perfect action is {best_arm}",
    "Clue: The prime action is {best_arm}",
    "Pointer: The superior action is {best_arm}",
    "Indication: The top action is {best_arm}",
    "Inference: The preeminent action is {best_arm}",
    "Lead: The supreme action is {best_arm}",
    "Cue: The paramount action is {best_arm}",
    "Guidance: The foremost action is {best_arm}",
    "Direction: The ultimate action is {best_arm}",
    "Instruction: The finest action is {best_arm}",
    "Recommendation: The exceptional action is {best_arm}",
    "Proposal: The outstanding action is {best_arm}",
    "Idea: The first-rate action is {best_arm}",
    "Prompt: The matchless action is {best_arm}",
    "Insinuation: The peerless action is {best_arm}",
    "Implication: The unrivaled action is {best_arm}",
    "Allusion: The unparalleled action is {best_arm}",
)

r_feedback = (
    "You received a reward of {reward}.",
    "You have been awarded {reward}.",
    "You've been given a reward of {reward}.",
    "You've earned a reward of {reward}.",
    "You've obtained a reward of {reward}.",
    "You've been rewarded with {reward}.",
    "You've been granted a reward of {reward}.",
    "You've been bestowed a reward of {reward}.",
    "You've been presented with a reward of {reward}.",
    "You've been conferred a reward of {reward}.",
    "You've been gifted a reward of {reward}.",
    "You've been accorded a reward of {reward}.",
    "You've been provided with a reward of {reward}.",
    "You've been endowed with a reward of {reward}.",
    "You've been given the reward of {reward}.",
    "You've been honored with a reward of {reward}.",
    "You've been acknowledged with a reward of {reward}.",
    "You've been recognized with a reward of {reward}.",
    "You've been appreciated with a reward of {reward}.",
    "You've been praised with a reward of {reward}.",
    "You've been credited with a reward of {reward}.",
)

hp_feedback = (
    "This is the best arm, as it has the highest expected reward.",
    "This arm is superior, given that it possesses the maximum anticipated reward.",
    "This is the top arm, since it carries the greatest expected payoff.",
    "This is the prime arm, as it holds the highest probable reward.",
    "This is the leading arm, as it contains the utmost expected prize.",
    "This is the foremost arm, as it has the supreme expected return.",
    "This is the preeminent arm, as it bears the highest likely reward.",
    "This is the paramount arm, as it has the maximal expected gain.",
    "This is the optimal arm, as it has the largest expected benefit.",
    "This is the unrivaled arm, as it has the most expected reward.",
    "This is the superior arm, as it has the highest potential reward.",
    "This is the finest arm, as it has the highest prospective reward.",
    "This is the most excellent arm, as it has the highest predicted reward.",
    "This is the outstanding arm, as it has the highest possible reward.",
    "This is the most favorable arm, as it has the highest anticipated reward.",
    "This is the most advantageous arm, as it has the highest expected reward.",
    "This is the most effective arm, as it has the highest expected reward.",
    "This is the most efficient arm, as it has the highest expected reward.",
    "This is the most superior arm, as it has the highest expected reward.",
    "This is the most exceptional arm, as it has the highest expected reward.",
    "This is the most distinguished arm, as it has the highest expected reward.",
)

hn_feedback = (
    "This is not the best arm, as it does not have the highest expected reward.",
    "This arm is not the best, as its expected reward is not the highest.",
    "The expected reward of this arm is not the highest, hence it is not the best.",
    "As the expected reward of this arm is not the highest, it is not the best.",
    "This arm is not the best because its expected reward is not the highest.",
    "This is not the top arm, as it doesn't have the maximum expected reward.",
    "This arm doesn't have the highest expected reward, so it's not the best.",
    "This arm, not having the highest expected reward, is not the best.",
    "This arm is not the best, given that it doesn't have the highest expected reward.",
    "This arm is not superior, as it does not possess the highest expected reward.",
    "This arm is not the best, as the expected reward is not the highest.",
    "This arm, which does not have the highest expected reward, is not the best.",
    "This arm is not the best, as it lacks the highest expected reward.",
    "This arm is not the best, as it falls short of the highest expected reward.",
    "This arm is not the best, as it does not yield the highest expected reward.",
    "This arm is not the best, as it does not command the highest expected reward.",
    "This arm is not the best, as it does not carry the highest expected reward.",
    "This arm is not the best, as it does not hold the highest expected reward.",
    "This arm is not the best, as it does not bear the highest expected reward.",
    "This arm is not the best, as it does not own the highest expected reward.",
    "This arm is not the best, as it does not maintain the highest expected reward.",
)

fp_feedback = (
    "You will receive an expected reward of {reward} if you choose action {best_arm}.",
    "An expected reward of {reward} will be received if you select action {best_arm}.",
    "If you opt for action {best_arm}, you'll get an expected reward of {reward}.",
    "Choosing action {best_arm} will yield an expected reward of {reward}.",
    "You'll be given an expected reward of {reward} if action {best_arm} is chosen.",
    "If action {best_arm} is your choice, the expected reward is {reward}.",
    "The expected reward for choosing action {best_arm} is {reward}.",
    "You are set to receive a reward of {reward} if you decide on action {best_arm}.",
    "An expected reward of {reward} is on the table if you pick action {best_arm}.",
    "You stand to gain an expected reward of {reward} if action {best_arm} is your pick.",
    "If you go with action {best_arm}, an expected reward of {reward} is yours.",
    "You're in for an expected reward of {reward} if you select action {best_arm}.",
    "Action {best_arm} will bring you an expected reward of {reward}.",
    "You'll earn an expected reward of {reward} if you choose action {best_arm}.",
    "If action {best_arm} is your selection, expect a reward of {reward}.",
    "You can anticipate a reward of {reward} if you opt for action {best_arm}.",
    "Choosing action {best_arm} will result in an expected reward of {reward}.",
    "If you decide for action {best_arm}, an expected reward of {reward} awaits.",
    "You'll be rewarded with {reward} if you take action {best_arm}.",
    "An expected reward of {reward} is what you'll get if you choose action {best_arm}.",
    "You're expected to receive a reward of {reward} if action {best_arm} is your choice.",
)

fn_feedback = (
    "Hint: Action {bad_action} is not the right one, as it gets an expected reward of {reward}."
    "Suggestion: The action {bad_action} is incorrect as it receives an expected reward of {reward}.",
    "Advice: The action {bad_action} is not appropriate since it yields an expected reward of {reward}.",
    "Note: The action {bad_action} is not suitable as it obtains an expected reward of {reward}.",
    "Tip: The action {bad_action} is not correct as it acquires an expected reward of {reward}.",
    "Pointer: The action {bad_action} is not the right choice as it gains an expected reward of {reward}.",
    "Recommendation: The action {bad_action} is not accurate as it secures an expected reward of {reward}.",
    "Guidance: The action {bad_action} is not proper as it fetches an expected reward of {reward}.",
    "Indication: The action {bad_action} is not the best as it earns an expected reward of {reward}.",
    "Inference: The action {bad_action} is not ideal as it receives an expected reward of {reward}.",
    "Insinuation: The action {bad_action} is not the correct one, as it results in an expected reward of {reward}.",
    "Implication: The action {bad_action} is not the right move as it gets an expected reward of {reward}.",
    "Suggestion: The action {bad_action} is not the right step as it brings in an expected reward of {reward}.",
    "Proposal: The action {bad_action} is not the right decision as it pulls in an expected reward of {reward}.",
    "Observation: The action {bad_action} is not the right course as it draws an expected reward of {reward}.",
    "Insight: The action {bad_action} is not the right path as it attracts an expected reward of {reward}.",
    "Counsel: The action {bad_action} is not the right method as it procures an expected reward of {reward}.",
    "Direction: The action {bad_action} is not the right approach as it achieves an expected reward of {reward}.",
    "Instruction: The action {bad_action} is not the right tactic as it accumulates an expected reward of {reward}.",
    "Guideline: The action {bad_action} is not the right strategy as it amasses an expected reward of {reward}.",
    "Clue: The action {bad_action} is not the right procedure as it collects an expected reward of {reward}.",
)