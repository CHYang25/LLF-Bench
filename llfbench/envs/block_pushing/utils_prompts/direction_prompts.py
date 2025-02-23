import random
import numpy as np

direction_desc_dict = {
    "x-axis": {
        "pos": ["rightward", "to the right", "toward the right"],
        "neg": ["leftward", "to the left", "toward the left"]
    },
    "y-axis": {
        "pos": ["to the front", "toward the front"],
        "neg": ["to the back", "toward the back"]
    },
}

direction_desc_list = [
    [
        ["rightward", "to the right", "toward the right"], 
        ["leftward", "to the left", "toward the left"]
    ],
    [        
        ["to the front", "toward the front"],
        ["to the back", "toward the back"]
    ],
]


def direction_converter(difference: np.array):
    return [
        random.choice(direction_desc_list[i][int(difference[i] < 0)])
        for i in range(2)
    ]