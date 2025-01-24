#!/usr/bin/env python
# coding: utf-8

# In[1]:


s1 = {1,2,3,4}
s2 = {3,4,5,6}
s1 | s2


# In[2]:


s1.intersection(s2)


# In[3]:


s1 & s2


# In[4]:


s1 = {2,3,5,6,7}
s2 = {5,6,7}
s1 - s2


# In[5]:


s2 - s1


# In[8]:


s1={1,2,3,4,5}
s2={4,5,6,7,8}
s1.symmetric_difference(s2)


# In[9]:


s2.symmetric_difference(s1)


# In[10]:


str1 = "Welcome to aiml class"
print(str1)
str2 = 'We started with python'
print(str2)
str3 = '''This is an awesome class'''
print(str3)


# In[11]:


print(type(str1))
print(type(str2))
print(type(str3))


# In[17]:


str4 = '''He said,"It's awesome!"'''
print(str4)


# In[18]:


print(str1)
str1[5:10]


# In[21]:


str1[::-1]


# In[22]:


dir(str)


# In[23]:


print(str1)
str1.split()


# In[24]:


reviews = ["The product is awesome", "Great service"]
joined_string = ' '.join(reviews)
joined_string


# In[25]:


str5 = "  Hello,How are you?  "
print(str5)


# In[31]:


str5.strip()


# In[32]:


d1 = {"lucky": 120, "mohan":150, "das":126}
for k in d1.keys():
    print(k)


# In[33]:


for v in d1.values():
    print(v)


# In[34]:


for k,v in d1.items():
    print(k,v)


# In[36]:


d1["luckkuuuu"] = 165
d1


# In[1]:


def avg_value(*n):
    l = len(n)
    average = sum(n)/1
    return average
avg_value(10,20,60100,900)


# In[3]:


greet = lambda name : print(f" Good Morning {name}!")
greet("Luckuu")


# In[4]:


#product of three numbers
product = lambda a,b,c : a*b*c
product(20,30,40)


# In[5]:


#lambda functions with list comprehension 
even = lambda L : [x for x in L if x%2 ==0]
my_list = [100,3,9,38,43,56,20]
even(my_list)


# In[ ]:




