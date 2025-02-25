import random

recommend_templates = [
    "move {direction} {degree}.",
    "please move {direction} {degree}.",
    "you should move {direction} {degree}.",
    "ensure you move {direction} {degree}.",
    "it is recommended to move {direction} {degree}.",
    "make sure to move {direction} {degree}.",
    "be sure to move {direction} {degree}.",
]


recommend_templates_sampler = lambda: random.choice(recommend_templates)