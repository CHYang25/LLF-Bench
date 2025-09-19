import random

recommend_templates = [
    "move {direction} {degree}.",
    "please move {direction} {degree}.",
    "you should move {direction} {degree}.",
    "make a move {direction} {degree}.",
    "you need to move {direction} {degree}.",
    "take a move {direction} {degree}.",
]


recommend_templates_sampler = lambda: random.choice(recommend_templates)