# exercise 3
# author: julian

import math
import random

# guess a  number

a = int(raw_input('Guess a number between 0 and 100 please!'))

# random number integer generation

x = random.randint(0, 100)
print(x)

# answer depending on input#
while x != a:
    print
    if a < x:
        print('Your guess is too low.')
        a = int(raw_input('Guess another number between 0 and 100 please!'))
    elif a > x:
        print('Your guess is too high.')
        a = int(raw_input('Guess another number between 0 and 100 please!'))
    else:
        print('You guessed the exact right number!')
        break
    print
