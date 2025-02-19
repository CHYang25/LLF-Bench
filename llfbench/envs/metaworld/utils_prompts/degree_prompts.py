import random
import numpy as np

degree_adverbs = {
    "very_low": ["gently", "softly", "quietly", "lightly", "carefully", "calmly", "hesitantly", "tentatively"],
    "low": ["steadily", "smoothly", "moderately", "casually", "slowly", "reluctantly"],
    "medium": ["firmly", "strongly", "vigorously", "confidently", "readily", "briskly", "eagerly"],
    "high": ["quickly", "intensely", "forcefully", "fiercely", "energetically", "boldly", "rapidly"],
    "very_high": ["aggressively", "powerfully", "recklessly", "relentlessly", "wildly", "tirelessly"],
    "highest": ["completely", "entirely", "absolutely", "utterly", "perfectly", "flawlessly"]
}

def degree_adverb_converter(difference: np.array):
    degrees = []
    for value in difference:
        value = abs(value)

        if value >= 0 and value < 1e-2:
            desc = random.choice(degree_adverbs["very_low"])
        elif value >= 1e-2 and value < 3e-2:
            desc = random.choice(degree_adverbs["low"])
        elif value >= 3e-2 and value < 5e-2:
            desc = random.choice(degree_adverbs["medium"])
        elif value >= 5e-2 and value < 8e-2:
            desc = random.choice(degree_adverbs["high"])
        elif value >= 8e-2 and value < 1e-1:
            desc = random.choice(degree_adverbs["very_high"])
        elif value >= 1e-1:
            desc = random.choice(degree_adverbs["highest"])
        
        degrees.append(desc)
    return degrees