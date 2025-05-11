import random
import numpy as np

direction_desc_dict = {
    "x-axis": {
        "pos": ["rightward", "to the right", "toward the right"],
        "neg": ["leftward", "to the left", "toward the left"]
    },
    "y-axis": {
        "pos": ["forward", "to the front", "toward the front"],
        "neg": ["backward", "to the back", "toward the back"]
    },
}

direction_desc_list = [
    [
        ["leftward", "to the left", "toward the left"],
        ["rightward", "to the right", "toward the right"] 
    ],
    [        
        ["forward", "to the front", "toward the front"],
        ["backward", "to the back", "toward the back"]
    ],
]

def direction_converter(difference: np.array):
    # TODO: need to check the difference
    return [
        random.choice(direction_desc_list[i][int(difference[i] < 0)])
        for i in range(2)
    ]