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

recorded_differences = []

def move_degree_adverb_converter(difference: np.array):
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

def turn_degree_adverb_converter(difference: np.array):
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

import matplotlib.pyplot as plt

def plot_difference_distribution():
    global recorded_differences
    data = np.array(recorded_differences)

    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Rotational Difference (radians)")
    plt.ylabel("Count")
    plt.title("Distribution of Rotational Differences")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig('./media/value_distribution.png')
