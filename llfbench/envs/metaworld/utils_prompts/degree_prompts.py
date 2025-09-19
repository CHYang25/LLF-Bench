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
        "slowly",
        "softly",
        "gently",
    ],
    "low": [
        "steadily",
        "calmly",
        "gradually",
    ],
    "medium": [
        "firmly",
        "quickly",
        "strongly",
    ]
}

def degree_adverb_converter(difference: np.array):
    degrees = []
    for value in difference:
        value = abs(value)

        if value >= 0 and value < 1e-2:
            desc = random.choice(degree_adverbs["very_low"])
        elif value >= 1e-2 and value < 3e-2:
            desc = random.choice(degree_adverbs["low"])
        elif value >= 3e-2:
            desc = random.choice(degree_adverbs["medium"])
        # elif value >= 5e-2 and value < 8e-2:
        #     desc = random.choice(degree_adverbs["high"])
        # elif value >= 8e-2 and value < 1e-1:
        #     desc = random.choice(degree_adverbs["very_high"])
        # elif value >= 1e-1:
        #     desc = random.choice(degree_adverbs["highest"])
        
        degrees.append(desc)
    return degrees