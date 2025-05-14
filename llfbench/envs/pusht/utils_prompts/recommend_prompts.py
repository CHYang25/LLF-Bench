import random

move_recommend_templates = [
    "the block should move {direction} {degree}.",
    "the block should move {direction} {degree}.",
    "the block should move {direction} {degree}.",
    "ensure the block to move {direction} {degree}.",
    "the block is recommended to move {direction} {degree}.",
    "make sure to let the block move {direction} {degree}.",
    "be sure to let the block move {direction} {degree}.",
    "try to move the block {direction} {degree}.",
    "consider moving the block {direction} {degree}.",
    "proceed to move the block {direction} {degree}.",
    "adjust by moving the block {direction} {degree}.",
    "take care to move the block {direction} {degree}.",
    "opt to move the block {direction} {degree}.",
    "plan to move the block {direction} {degree}.",
    "execute a move the block {direction} {degree}.",
    "aim to move the block {direction} {degree}.",
]

turn_recommend_templates = [
    "the block should turn {direction} {degree}.",
    "try to turn the block {direction} {degree}.",
    "it is a good idea to turn the block {direction} {degree}.",
    "you might want to turn the block {direction} {degree}.",
    "it would be smart to turn the block {direction} {degree}.",
    "turning the block {direction} {degree} could work well.",
    "you could aim to turn the block {direction} {degree}.",
    "it would help to turn the block {direction} {degree}.",
]

move_recommend_templates_sampler = lambda: random.choice(move_recommend_templates)
turn_recommend_templates_sampler = lambda: random.choice(turn_recommend_templates)