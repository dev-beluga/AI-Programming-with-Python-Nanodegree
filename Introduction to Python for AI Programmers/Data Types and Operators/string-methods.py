print("torpa fekes".title())

test_str = "one love"

print(test_str.islower())
print(test_str.count('o'))
print(test_str.find('v'))  
 
print("Mohammed has {} balloons".format(27)) 
animal = "dog" 
action = "bite" 
print("Does your {} {}?".format(animal, action)) 
maria_string = "Maria loves {} and {}" 
print(maria_string.format("math", "statistics")) 

new_str = "The cow jumped over the moon."

print(new_str.split())

# 2. Here  the separator is space, and the maxsplit argument is set to 3.
print(new_str.split(' ', 3))
  
# 3. Using '.' or period as a separator.
print(new_str.split('.'))

# 4. Using no separators but having a maxsplit argument of 3.
print(new_str.split(None, 3))

verse = "If you can keep your head when all about you\n  Are losing theirs and blaming it on you,\nIf you can trust yourself when all men doubt you,\n  But make allowance for their doubting too;\nIf you can wait and not be tired by waiting,\n  Or being lied about, don’t deal in lies,\nOr being hated, don’t give way to hating,\n  And yet don’t look too good, nor talk too wise:"
# print(verse, "\n")

print("Verse has a length of {} characters.".format(len(verse)))
print("The first occurence of the word 'and' occurs at the {}th index.".format(verse.find('and')))
print("The last occurence of the word 'you' occurs at the {}th index.".format(verse.rfind('you')))
print("The word 'you' occurs {} times in the verse.".format(verse.count('you')))
