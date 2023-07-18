# Lists and slice adn dice methodsvv  mutable and ordered
list_of_random_things = [1, 3.4, 'a string', True]
""" print(list_of_random_things[1:2])
print(list_of_random_things[:2])
print(list_of_random_things[1:])
"""
# String ordered and immutable

# in and not in  -- return a bool
""" print('this' in 'this is a string')
print('isa' in 'this is a string') """

# Tuples ordered and immutable
tuple_a= 1,2
tuple_b= (1,2)
# print(type(list_of_random_things))
# print(type(tuple_a))
# print(tuple_a == tuple_b)
# print(tuple_b.)

# Sets  -- mutable  and unordered data type {}
numbers=[1,23,2,3,2,1]
num=dict()
unique_nums=set(numbers)
print(type(num))
print((num))
print(len(unique_nums)) 

"""
a = [1, 2, 3]
b = a
c = [1, 2, 3]

print(a == b) #T
print(a is b) #T
print(a == c)#T
print(a is c)#F

"""
# Dictionaries -- mutable and unordered
animals = {'dogs': [20, 10, 15, 8, 32, 15], ('cats','meow'): [3,4,2,8,2,4], 'rabbits': [2, 3, 3], 'fish': [0.3, 0.5, 0.8, 0.3, 1]}
print(animals[('cats','meow')])