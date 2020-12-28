#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[1]:


import numpy as np
arr = np.array([0,1,2,3,4,5,6,7,8,9])
np.reshape(arr, (2,5))


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[11]:


arr1 = np.array([[0,1,2,3,4],[5,6,7,8,9]])
arr2 = np.array([[1,1,1,1,1],[1,1,1,1,1]])
np.vstack((arr1, arr2))


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[12]:


arr1 = np.array([[0,1,2,3,4],[5,6,7,8,9]])
arr2 = np.array([[1,1,1,1,1],[1,1,1,1,1]])
np.hstack((arr1, arr2))


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[19]:


arr3 = np.array([[0,1,2,3,4],[5,6,7,8,9]])
arr3.reshape([1,10])


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[28]:


arr4 = np.array([[[[0,1,2,3,4]],[[5,6,7,8,9]],[[10,11,12,13,14]]]])
arr4.ndim
arr4.reshape([1, 15])


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[36]:


arr5  = np.arange(15)
arr6  = arr5.reshape(5, 3)
print(arr6, arr6.ndim)

arr7 = arr5.reshape(3, 1, 5)
print(arr7, arr7.ndim)


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[23]:


arr5d = np.array([[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24]])
print("Square of the array")
np.square(arr5d)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[9]:


array = np.array([[0,1,2,3,4,5],[6,7,8,9,10,11],[12,13,14,15,16,17],[18,19,20,21,22,23],[24,25,26,27,28,29]])
array.shape
np.mean(array)


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[21]:


print("standard deviation of the array which is defined in Q8")
np.std(array)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[20]:


print("median of the array which is defined in Q8")
np.median(array)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[19]:


print("transpose of the array which is defined in Q8")
np.transpose(array)


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[43]:


array4 = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
array4.shape
#by using diagonal method
diagnl = np.diagonal(array4)
print("Diagonal elements are : ", diagnl)
print("sum of the diagnl elements are: ", sum(diagnl))
#by using trace
print("by using trace method:", np.trace(array4))


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[27]:


print(array4)
np.linalg.det(array4)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[31]:


Parr = np.array([1,13,25,37,49])
print("5th percentile of array : ", 
       np.percentile(Parr, 5))
print("95th percentile of array : ", 
       np.percentile(Parr, 95))


# ## Question:15

# ### How to find if a given array has any null values?

# In[44]:


arraychk = np.array([0,np.nan,2,3,4,5,6,7,8,9])
np.isnan(arraychk)


# In[ ]:




