import random

positive_conjunctions = [
    " Also, ", " Moreover, ", " Furthermore, ", 
    " In addition, ", " Plus, ", 
]

negative_conjunctions = [
    " However, ", " But, ",
    " Nonetheless, ", " Even so, ", " Despite that, "
]

positive_conjunctions_sampler = lambda: random.choice(positive_conjunctions)
negative_conjunctions_sampler = lambda: random.choice(negative_conjunctions)