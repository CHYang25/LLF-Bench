import random
import numpy as np

move_direction_desc_dict = {
    "x-axis": {
        "pos": ["rightward", "to the right", "toward the right"],
        "neg": ["leftward", "to the left", "toward the left"]
    },
    "y-axis": {
        "pos": ["forward", "to the front", "toward the front"],
        "neg": ["backward", "to the back", "toward the back"]
    },
    "z-axis": {
        "pos": ["upward", "up", "to the top"],
        "neg": ["downward", "down", "to the bottom"]
    }
}

move_direction_desc_list = [
    [
        ["leftward", "to the left", "toward the left"],
        ["rightward", "to the right", "toward the right"] 
    ],
    [        
        ["forward", "to the front", "toward the front"],
        ["backward", "to the back", "toward the back"]
    ],
    [
        ["upward", "up", "to the top"], 
        ["downward", "down", "to the bottom"]
    ]
]


def move_direction_converter(difference: np.array):
    return [
        random.choice(move_direction_desc_list[i][int(difference[i] < 0)])
        for i in range(3)
    ]

turn_direction_desc_dict = {
    "x-axis": {
        "pos": ["pitch up", "pitch to the top", "pitch toward the top"],
        "neg": ["pitch down", "pitch to the bottom", "pitch toward the bottom"]
    },
    "y-axis": {
        "pos": ["roll left", "roll to the left", "roll toward the left"],
        "neg": ["roll right", "roll to the right", "roll toward the right"]
    },
    "z-axis": {
        "pos": ["yaw left", "yaw to the left", "yaw toward the left"],
        "neg": ["yaw right", "yaw to the right", "yaw toward the right"]
    }
}

turn_direction_desc_list = [
    [
        ["pitch down", "pitch to the bottom", "pitch toward the bottom"],
        ["pitch up", "pitch to the top", "pitch toward the top"]
    ],
    [
        ["roll left", "roll to the left", "roll toward the left"],
        ["roll right", "roll to the right", "roll toward the right"]
    ],
    [
        ["yaw left", "yaw to the left", "yaw toward the left"],
        ["yaw right", "yaw to the right", "yaw toward the right"]
    ]
]

def turn_direction_converter(difference: np.array):
    return [
        random.choice(turn_direction_desc_list[i][int(difference[i] < 0)])
        for i in range(3)
    ]