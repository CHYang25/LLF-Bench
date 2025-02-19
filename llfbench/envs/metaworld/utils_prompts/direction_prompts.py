import random
import numpy as np

direction_desc_dict = {
    "x-axis": {
        "pos": ["rightward", "to the right", "toward the right"],
        "neg": ["leftward", "to the left", "toward the left"]
    },
    "y-axis": {
        "pos": ["forward", "to the front", "ahead", "toward the front"],
        "neg": ["backward", "to the back", "behind", "toward the back"]
    },
    "z-axis": {
        "pos": ["upward", "up", "to the top"],
        "neg": ["downward", "down", "to the bottom"]
    }
}

direction_desc_list = [
    [
        ["leftward", "to the left", "toward the left"],
        ["rightward", "to the right", "toward the right"] 
    ],
    [        
        ["forward", "to the front", "ahead", "toward the front"],
        ["backward", "to the back", "behind", "toward the back"]
    ],
    [
        ["upward", "up", "to the top"], 
        ["downward", "down", "to the bottom"]
    ]
]


def direction_converter(difference: np.array):
    return [
        random.choice(direction_desc_list[i][int(difference[i] < 0)])
        for i in range(3)
    ]