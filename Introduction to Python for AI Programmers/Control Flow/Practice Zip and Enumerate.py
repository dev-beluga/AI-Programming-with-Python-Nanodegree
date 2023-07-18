#!/usr/bin/env python
# coding: utf-8

# ## Zip and Enumerate
# 

# In[7]:


list(zip(['a', 'b', 'c'], [1, 2, 3]))


# In[8]:


letters = ['a', 'b', 'c']
nums = [1, 2, 3]

for letter, num in zip(letters, nums):
    print("{}: {}".format(letter, num))


# # In addition to zipping two lists together, you can also unzip a list into tuples using an asterisk.

# In[9]:


some_list = [('a', 1), ('b', 2), ('c', 3)]
letters, nums= zip(*some_list)
print(letters, nums)


# In[11]:


letters = ['a', 'b', 'c', 'd', 'e']
for i, letter in enumerate(letters):
    print(i, letter)


# In[ ]:




