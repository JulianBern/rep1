#exercises 1 and 2
#author: julian

import math

#input questions for user

a = raw_input('What is your full name?')
c = int(raw_input('How old are you?'))
g = (raw_input('Press f for female or m for male!'))

#calculations

d = 10 - (c%10)
e = int(d + c)

#answer depending on inputs

if g == 'f' and c >= 18:
    print('Mrs. ' + a + ", in " + str(d) + " years you will be " + str(e) + ".")
elif g == 'm' and c >= 18:
    print('Mr. ' + a + ", in " + str(d) + " years you will be " + str(e) + ".")
else:
    print(a + ", in " + str(d) + " years you will be " + str(e) + ".")
