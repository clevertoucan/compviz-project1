#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np, seaborn as sns, pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as img
import glob


# Step 1: Obtain face images I1, I2, ..., IM (training faces)
# 
# Step 2: Represent each image as a vector.

# In[14]:

path = r'./Train1' # use your path
all_images = glob.glob(path + "/*.jpg")

A = []

for image in all_images:
    arr = img.imread(image).flatten()
    A.append(arr)
    
A = np.asarray(A)
A


# Step 3: Compute the average face vector Ψ

# In[15]:


mean_vector = np.mean(A, axis=0, dtype = np.float64)
mean_vector


# Step 4: Subtract the mean face.

# In[16]:


A = np.subtract(A, mean_vector)
A


# Step 6: Get M best eigenvalues and eigenvectors of A * A^T

# In[7]:


A = A.T #redefine each image within the column
eig_vals, eig_vects = np.linalg.eigh(np.matmul(A.T, A))

# Step 7: Get K best eigenvalues and eigenvectors.
# For future people, each row corresponds to an eigenvector and its' eigenvalue at the end.

# In[8]:

#this is our K-value
k = 20
test = pd.DataFrame(eig_vects.T)
test2 = pd.Series(eig_vals)
test['Eigenvalues'] = test2
result = test.nlargest(k, ['Eigenvalues'], keep='first')
result = np.asarray(result)
eigenspace = []
for i in range(0, k):
    x = result[i][0:-1]
    x = np.matmul(A, x)
    eigenspace.append(x)


projections = []
for image in all_images:
    arr = img.imread(image).flatten()
    phi = arr - mean_vector
    sigma = []
    for i in range(0, k):
        w = np.matmul(np.transpose(eigenspace[i]), phi)
        sigma.append(w)
    projections.append(sigma)

test_path = r'./Test1'
test_images = glob.glob(path + "/*.jpg")
A = []

for test_image in test_images:
    arr = img.imread(test_image).flatten()
    phi = arr - mean_vector
    sigma = []
    for i in range(0, k):
        w = np.matmul(np.transpose(eigenspace[i]), phi)
        sigma.append(w)
    x = 0
    minimum = 1000000
    min_index = -1
    for train_image in all_images:
        similarity = np.linalg.norm(np.subtract(sigma, projections[x]))
        if similarity < minimum:
            minimum = similarity
            min_index = x

    print(minimum)
    print(min_index)

# In[ ]:




