#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Quiz: Write a Docstring
def readable_timedelta(days):
    """ NUMBER OF WEEKS AND DAYS FOR A GIVEN LENGTH OF DAYS """    
    weeks = days // 7
    remainder = days % 7
    return "{} week(s) and {} day(s)".format(weeks, remainder)
    


# In[ ]:


def readable_timedelta(days):
    """Return a string of the number of weeks and days included in days."""
    weeks = days // 7
    remainder = days % 7
    return "{} week(s) and {} day(s)".format(weeks, remainder)
def readable_timedelta(days):
    """Return a string of the number of weeks and days included in days.

    Args:
        days (int): number of days to convert
    """
    weeks = days // 7
    remainder = days % 7
    return "{} week(s) and {} day(s)".format(weeks, remainder)
def readable_timedelta(days):
    """
    Return a string of the number of weeks and days included in days.

    Parameters:
    days -- number of days to convert (int)

    Returns:
    string of the number of weeks and days included in days
    """
    weeks = days // 7
    remainder = days % 7
    return "{} week(s) and {} day(s)".format(weeks, remainder)

