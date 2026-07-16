import random

positive_conjunctions = [
    " Also, ", " Moreover, ", " Furthermore, ", " Plus, ", 
]

progressive_conjunctions = [
    " Then, ", " Next, ", " Afterward, ", " After that, ",
]

negative_conjunctions = [
    " However, ", " But, ", " Yet, ", " Even so, ",
]

positive_conjunctions_sampler = lambda: random.choice(positive_conjunctions)
progressive_conjunction_sampler = lambda: random.choice(progressive_conjunctions)
negative_conjunctions_sampler = lambda: random.choice(negative_conjunctions)