import random
import numpy as np

move_direction_desc_dict = {
    "x-axis": {
        "pos": ["to the east", "toward the east"],
        "neg": ["to the west", "toward the west"]
    },
    "y-axis": {
        "pos": ["to the north", "toward the north"],
        "neg": ["to the south", "toward the south"]
    },
}

move_direction_desc_list = [
    [
        ["to the west", "toward the west"],
        ["to the east", "toward the east"] 
    ],
    [        
        ["to the north", "toward the north"],
        ["to the south", "toward the south"]
    ],
]

turn_direction_desc_dict = {
    "angle": {
        "pos": ["clockwise", "rightward around the center", "with positive angle"],
        "neg":  ["counterclockwise", "leftward around the center", "with negative angle"],
    }
}

turn_direction_desc_list = [
    [
        ["counterclockwise", "against the direction of a clock"],
        ["clockwise", "in the direction of a clock"],
    ]
]

def move_direction_converter(difference: np.array):
    return [
        random.choice(move_direction_desc_list[i][int(difference[i] > 0)])
        if difference[i] != 0 else ""
        for i in range(difference.shape[0])
    ]

def turn_direction_converter(difference: float):
    if difference == 0:
        return ""
    return random.choice(turn_direction_desc_list[0][int(difference > 0)])