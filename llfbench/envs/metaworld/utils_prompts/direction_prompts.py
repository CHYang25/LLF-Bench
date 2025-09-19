import random
import numpy as np

direction_desc_dict = {
    "x-axis": {
        "pos": ["to the right", "toward the right"],
        "neg": ["to the left", "toward the left"]
    },
    "y-axis": {
        "pos": ["to the front", "toward the front"],
        "neg": ["to the back", "toward the back"]
    },
    "z-axis": {
        "pos": ["upward", "to the top"],
        "neg": ["downward", "to the bottom"]
    }
}

direction_desc_list = [
    [
        ["to the left", "toward the left"],
        ["to the right", "toward the right"] 
    ],
    [        
        ["to the front", "toward the front"],
        ["to the back", "toward the back"]
    ],
    [
        ["upward", "to the top"], 
        ["downward", "to the bottom"]
    ]
]


def direction_converter(difference: np.array):
    return [
        random.choice(direction_desc_list[i][int(difference[i] < 0)])
        for i in range(3)
    ]