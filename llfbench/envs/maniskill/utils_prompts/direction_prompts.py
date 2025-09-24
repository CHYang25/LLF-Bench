import random
import numpy as np

move_direction_desc_dict = {
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

move_direction_desc_list = [
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


def move_direction_converter(difference: np.array):
    return [
        random.choice(move_direction_desc_list[i][int(difference[i] < 0)])
        for i in range(3)
    ]

turn_direction_desc_dict = {
    "x-axis": {
        "pos": ["pitch to the top", "pitch toward the top"],
        "neg": ["pitch to the bottom", "pitch toward the bottom"]
    },
    "y-axis": {
        "pos": ["roll to the left", "roll toward the left"],
        "neg": ["roll to the right", "roll toward the right"]
    },
    "z-axis": {
        "pos": ["yaw to the left", "yaw toward the left"],
        "neg": ["yaw to the right", "yaw toward the right"]
    }
}

turn_direction_desc_list = [
    [
        ["pitch to the bottom", "pitch toward the bottom"],
        ["pitch to the top", "pitch toward the top"]
    ],
    [
        ["roll to the left", "roll toward the left"],
        ["roll to the right", "roll toward the right"]
    ],
    [
        ["yaw to the left", "yaw toward the left"],
        ["yaw to the right", "yaw toward the right"]
    ]
]

def turn_direction_converter(difference: np.array):
    return [
        random.choice(turn_direction_desc_list[i][int(difference[i] < 0)])
        for i in range(3)
    ]