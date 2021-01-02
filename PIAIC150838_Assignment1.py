#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[2]:


data = np.zeros(10)


# 3. Create a vector with values ranging from 10 to 49

# In[3]:


data1 = np.arange(10,49)


# 4. Find the shape of previous array in question 3

# In[4]:


data1.shape


# 5. Print the type of the previous array in question 3

# In[5]:


data1.dtype


# 6. Print the numpy version and the configuration
# 

# In[6]:


print(np.__version__)


# 7. Print the dimension of the array in question 3
# 

# In[7]:


data1.ndim


# 8. Create a boolean array with all the True values

# In[14]:


boolean_arr = np.ones( 10 , dtype = bool)
boolean_arr


# 9. Create a two dimensional array
# 
# 
# 

# In[10]:


array1 = np.array([[2,2], [3,3]])
array1.ndim


# 10. Create a three dimensional array
# 
# 

# In[11]:


array1 = np.array([[1,1,1],[2,2,2], [3,3,3]])


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[16]:


data2 = np.arange(1, 20)
print("Original array:")
print (data2)
print("Reverse array:")
data2 = data2[::-1]
data2


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[18]:


arr1 = np.zeros(10)
print(arr1)
print("Update fifth value to 1")
arr1[4] = 1
print(arr1)


# 13. Create a 3x3 identity matrix

# In[19]:


arr2 = np.identity(3)
print(arr2)


# In[30]:


arr = np.array([1, 2, 3, 4, 5])
print(arr.dtype)
print("after converting the data type ")
arr = arr.astype('float64')
arr.dtype


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[34]:


import numpy as np
arr1 = np.array([[1., 2., 3.],
                 
                 [4., 5., 6.]])  

arr2 = np.array([[0., 4., 1.],
                 
                 [7., 2., 12.]])
arr3 = np.multiply(arr1, arr2)
arr3


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[36]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])
arr3 = (arr1 == arr2)
arr3


# 17. Extract all odd numbers from arr with values(0-9)

# In[47]:


arr = np.array([1,2,3,5,9,6,3,4,5,6,6,7,7,8,9])
arr[arr%2 ==1]


# 18. Replace all odd numbers to -1 from previous array

# In[52]:


arr = np.array([1,2,3,5,9,6,3,4,5,6,6,7,7,8,9])
arr[arr%2 ==1] = -1
arr


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[12]:


import numpy as np
arr = np.arange(10)
arr[5:8] = 12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[72]:


arr = np.ones((4,4))
arr[1:-1,1:-1] = 0
arr


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[109]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
arr2d[1][1] = 12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[111]:


arr3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
arr3d[0] = 64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[125]:


arr2d = np.array([[1,3,4,2,5],[6,7,8,9,0]])
print(arr2d.ndim)
arr2d[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[126]:


arr2d = np.array([[1,3,4,2,5],[6,7,8,9,0]])
print(arr2d.ndim)
arr2d[1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[154]:


arr = np.array([[0,1,2,3,4],[5,6,7,8,9]])
arr[0:2, 2:3]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[161]:


arr = np.random.randn(10,10)
print(arr)
print(f"minimum value of the array is {arr.min()} ")
print(f"maximum value of the array is {arr.max()} ")


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[162]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a,b))


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[166]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.where(a==b))


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[210]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
x = data[names != "Will"] 
x


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[214]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
x = data[names != ("Will" and "Joe")]
x


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[174]:


arr = np.array([[1,2,3],[4,5,6],[7,8,9],[11,12,13],[14,15,0]])
print(arr.shape)
print(arr.ndim)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[180]:


arr2 = np.array([[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]]])
arr2.shape


# 33. Swap axes of the array you created in Question 32

# In[194]:


arr2 = np.array([[[1,2,3,4],[5,6,7,8]],[[9,10,11,12],[13,14,15,16]]])
np.swapaxes(arr2,0,2)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[222]:


arr3 = np.arange(10)
np.sqrt(arr3)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[234]:


arr4 = np.random.randn(12)
arr5 = np.random.randn(12)
print(arr4)
print(arr5)
np.maximum.reduce([arr4,arr5])


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[246]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.sort(np.unique(names))


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[247]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
np.setdiff1d(a, b)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[30]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
print(sampleArray[0:3])
sampleArray = np.delete(sampleArray, 1 , 1)
print(sampleArray)
newColumn = np.array([[10,10,10]])
NewArr = np.insert(sampleArray,1, newColumn, axis = 1 )
print(NewArr)


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[248]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[262]:


matarr = np.random.randn(20,20)
print(matarr)
matarr.cumsum()


# In[ ]:




