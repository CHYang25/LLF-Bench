import random

positive_conjunctions = [
    " Also, ", " Additionally, ", " Moreover, ", " Furthermore, ", 
    " In addition, ", " Plus, ", 
]

negative_conjunctions = [
    " However, ", " But, ", " Yet, ", 
    " Nevertheless, ", " Nonetheless, ", " Even so, ", " Despite that, "
]

positive_conjunctions_sampler = lambda: random.choice(positive_conjunctions)
negative_conjunctions_sampler = lambda: random.choice(negative_conjunctions)