#!/usr/bin/env python
# coding: utf-8

# In[2]:


card_deck = [4, 11, 8, 5, 13, 2, 8, 10]
hand = []

## adds the last element of the card_deck list to the hand list
## until the values in hand add up to 17 or more
while sum(hand)  < 17:
    hand.append(card_deck.pop())
    
print(hand)


# # QUIZ: Count By
# 

# In[3]:


start_num = 5 #provide some start number, replace 5 with a number you choose
end_num = 100#provide some end number that you stop when you hit, replace 100 with a number you choose
count_by = 2 #provide some number to count by, replace 2 with a number you choose 

# write a while loop that uses break_num as the ongoing number to 
# check against end_num
break_num = start_num #replace None with appropriate code
while break_num < end_num:
#     print(break_num)
    break_num+=count_by

print(break_num)


# # Quiz: Count By Check

# In[7]:


start_num = 5 #provide some start number, replace 5 with a number you choose
end_num = 100#provide some end number that you stop when you hit, replace 100 with a number you choose
count_by = 2 #provide some number to count by, replace 2 with a number you choose 

# write a condition to check that end_num is larger than start_num before looping
# write a while loop that uses break_num as the ongoing number to 
# check against end_num
result = None #replace None with appropriate code
break_num = start_num #replace None with appropriate code
if start_num < end_num:
    while break_num < end_num:
        break_num+=count_by
        result=break_num
else:
    result="Oops!  Looks like your start value is greater than the end value.  Please try again."

print(break_num, result)


# ## Quiz: Nearest Square
# 

# In[37]:


limit = 40 #provide a limit, replace 40 with a number you choose
# write your while loop here
number=1
while number**2<limit:
    nearest_square=number**2
    number+=1 
print(nearest_square)


# In[34]:


limit = 40

num = 0
while (num+1)**2 < limit:
    num += 1
    nearest_square = num**2

print(nearest_square)


# In[ ]:




