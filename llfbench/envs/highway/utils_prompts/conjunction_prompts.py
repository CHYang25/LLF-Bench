import random

positive_conjunctions = [
    " Also, ", " Moreover, ", " Furthermore, ", " Plus, ", 
]

negative_conjunctions = [
    " However, ", " But, ", " Yet, ", " Even so, ",
]

positive_conjunctions_sampler = lambda: random.choice(positive_conjunctions)
negative_conjunctions_sampler = lambda: random.choice(negative_conjunctions)