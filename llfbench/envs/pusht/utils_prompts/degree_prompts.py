import random
import numpy as np

# degree_adverbs = {
#     "very_low": ["gently", "softly", "carefully"],
#     "low": ["steadily", "smoothly", "slowly"],
#     "medium": ["firmly", "strongly", "confidently"],
#     # "high": ["quickly", "intensely", "forcefully", "fiercely", "energetically", "boldly", "rapidly"],
#     # "very_high": ["aggressively", "powerfully", "recklessly", "relentlessly", "wildly", "tirelessly"],
#     # "highest": ["completely", "entirely", "absolutely", "utterly", "perfectly", "flawlessly"]
# }

degree_adverbs = {
    "very_low": [
        "gently",          # Minimal force, soft touch
        "softly",          # Light pressure, subtle motion
        "carefully",       # Precise, cautious adjustment
        "lightly",         # Barely perceptible force
        "delicately",      # Fine, fragile handling
        "precisely",       # Exact, minimal deviation
        "cautiously",      # Slow, deliberate care
        "subtly",          # Small, nuanced shift
        "tentatively",     # Hesitant, testing motion
        "finely",          # Tiny, controlled tweak
        "gingerly"         # Extra care to avoid damage
    ],
    "low": [
        "steadily",        # Consistent, even pace
        "smoothly",        # Fluid, uninterrupted motion
        "slowly",          # Reduced speed, controlled
        "evenly",          # Uniform, balanced movement
        "gradually",       # Incremental, step-by-step
        "calmly",          # Relaxed, steady progress
        "consistently",    # Regular, predictable rate
        "leisurely",       # Unrushed, moderate pace
        "methodically",    # Systematic, planned motion
        "patiently",       # Deliberate, unhurried
        "uniformly"        # Equal, steady application
    ],
    "medium": [
        "firmly",          # Solid, confident force
        "strongly",        # Notable strength, assured
        "confidently",     # Bold, self-assured motion
        "solidly",         # Stable, grounded movement
        "boldly",          # Assertive, decisive action
        "briskly",         # Quick, energetic shift
        "stably",          # Steady with moderate force
        "vigorously",      # Active, forceful motion
        "decisively",      # Clear, purposeful action
        "securely",        # Tight, reliable grip or move
        "actively"         # Engaged, dynamic motion
    ]
}

def move_degree_adverb_converter(difference: np.array):
    degrees = []
    for value in difference:
        value = abs(value)

        # window size = 512x512
        if value >= 0 and value < 10:
            desc = random.choice(degree_adverbs["very_low"])
        elif value >= 10 and value < 300:
            desc = random.choice(degree_adverbs["low"])
        elif value >= 300:
            desc = random.choice(degree_adverbs["medium"])
        # elif value >= 5e-2 and value < 8e-2:
        #     desc = random.choice(degree_adverbs["high"])
        # elif value >= 8e-2 and value < 1e-1:
        #     desc = random.choice(degree_adverbs["very_high"])
        # elif value >= 1e-1:
        #     desc = random.choice(degree_adverbs["highest"])
        
        degrees.append(desc)
    return degrees

def turn_degree_adverb_converter(difference: np.array):
    difference = abs(difference)

    if difference >= 0 and difference < 0.5:
        desc = random.choice(degree_adverbs["very_low"])
    elif difference >= 0.5 and difference < 1.57:
        desc = random.choice(degree_adverbs["low"])
    elif difference >= 1.57:
        desc = random.choice(degree_adverbs["medium"])
    # elif value >= 5e-2 and value < 8e-2:
    #     desc = random.choice(degree_adverbs["high"])
    # elif value >= 8e-2 and value < 1e-1:
    #     desc = random.choice(degree_adverbs["very_high"])
    # elif value >= 1e-1:
    #     desc = random.choice(degree_adverbs["highest"])
        
    return desc
