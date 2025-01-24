import random

positive_conjunctions = [
    " Also, ", " What's more, ", " Additionally, ", " Moreover, ", " Furthermore, ", 
    " In addition, ", " Not only that, ", " On top of that, ", " Plus, ", " As well, "
]

negative_conjunctions = [
    " However, ", " But, ", " On the other hand, ", " Yet, ", " Still, ", 
    " Nevertheless, ", " Nonetheless, ", " Even so, ", " Though, ", " Despite that, "
]

positive_conjunctions_sampler = lambda: random.choice(positive_conjunctions)
negative_conjunctions_sampler = lambda: random.choice(negative_conjunctions)