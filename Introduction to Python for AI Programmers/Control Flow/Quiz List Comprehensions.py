#!/usr/bin/env python
# coding: utf-8

# In[6]:


squares = [x**2 if x % 2 == 0 else x + 3 for x in range(9) ]
print(squares)


# In[12]:


names = ["Rick Sanchez", "Morty Smith", "Summer Smith", "Jerry Smith", "Beth Smith"]
first_names = [name.lower().split()[0] for name in names] # write your list comprehension here
print(first_names)


# In[2]:


multiples_3 = [x*3 for x in range(1,21)] # write your list comprehension here
print(multiples_3)


# In[16]:


scores = {
             "Rick Sanchez": 70,
             "Morty Smith": 35,
             "Summer Smith": 82,
             "Jerry Smith": 23,
             "Beth Smith": 98
          }

passed = [name for name, score in scores.items() if score >=65] # write your list comprehension here
print(passed)


# In[ ]:





# In[ ]:




