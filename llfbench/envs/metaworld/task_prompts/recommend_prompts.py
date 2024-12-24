import random

recommend_templates = [
    "you should move {direction} {degree}.",
    "it is recommended to move {direction} {degree}.",
    "consider moving {direction} {degree}.",
    "try moving {direction} {degree}.",
    "you may want to move {direction} {degree}.",
    "it's best to move {direction} {degree}.",
    "moving {direction} {degree} might help.",
    "you could move {direction} {degree}.",
    "it would be ideal to move {direction} {degree}.",
    "ensure you move {direction} {degree}.",
    "take action to move {direction} {degree}.",
    "a good step would be moving {direction} {degree}.",
]


recommend_templates_sampler = lambda: random.choice(recommend_templates)