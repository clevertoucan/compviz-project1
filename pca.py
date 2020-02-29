#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np, seaborn as sns, pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as img, time
import glob


arr = [(20, 0.5), (40, 0.5681818181818182), (50, 0.5681818181818182), (55, 0.5681818181818182),
       (60, 0.5909090909090909), (65, 0.5909090909090909), (70, 0.5909090909090909),
       (75, 0.5909090909090909), (80, 0.5909090909090909), (160, 0.5909090909090909),
       (320, 0.5909090909090909), (640, 0.5909090909090909), (800, 0.5909090909090909), (900, 0.5909090909090909),
       (1000, 0.5909090909090909), (1100, 0.5909090909090909), (1200, 0.5909090909090909), (1300, 0.5909090909090909),
       (1400, 0.5909090909090909), (1500, 0.5909090909090909), (3000, 0.5909090909090909)]

plt.scatter(*zip(*arr))
plt.show()

# Step 1: Obtain face images I1, I2, ..., IM (training faces)
# 
# Step 2: Represent each image as a vector.

# In[14]:
plot = []
#this is our K-value
k = 60
path = r'./Train1' # use your path
all_images = glob.glob(path + "/*.jpg")

A = []

for image in all_images:
    arr = img.imread(image).flatten()
    A.append(arr)

A = np.asarray(A)


# Step 3: Compute the average face vector Ψ

# In[15]:


mean_vector = np.mean(A, axis=0, dtype = np.float64)
mean_vector


# Step 4: Subtract the mean face.

# In[16]:


A = np.subtract(A, mean_vector)


# Step 6: Get M best eigenvalues and eigenvectors of A * A^T

# In[7]:


A = A.T #redefine each image within the column
eig_vals, eig_vects = np.linalg.eigh(np.matmul(A.T, A))

# Step 7: Get K best eigenvalues and eigenvectors.
# For future people, each row corresponds to an eigenvector and its' eigenvalue at the end.

# In[8]:
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


#fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 5
eigenfaces = []
i = 1
for vec in eigenspace:
    face = vec.reshape((150, 130))
    eigenfaces.append(face)
    image = face
    #fig.add_subplot(rows, columns, i)
    #plt.imshow(image)
    #plt.axis('off')
    i += 1

#plt.show()

#fig = plt.figure(figsize=(22, 5))
columns = k + 2
rows = 5
for i in range(0,5):
    image = img.imread(all_images[i])
    #fig.add_subplot(rows, columns, (i * columns) + 1)
    #plt.imshow(image)
    #plt.axis('off')
    sigma = projections[i]
    sum_faces = np.zeros((150, 130))
    for j in range(0, k):
        face = eigenfaces[j] * sigma[j]
        sum_faces += face
        #fig.add_subplot(rows, columns, (i * columns) + 1 + j + 1)
        #plt.imshow(face)
        #plt.axis('off')
    #fig.add_subplot(rows, columns, i * columns + columns)
    #plt.imshow(sum_faces)
    #plt.axis('off')
#plt.show()


test_path = r'./Test1'
test_images = glob.glob(test_path + "/*.jpg")
A = []

imgs = []
sample = 0
accuracy = 0.0
for test_image in test_images:
    name = test_image[8:]
    arr = img.imread(test_image).flatten()
    phi = arr - mean_vector
    sigma = []
    for i in range(0, k):
        w = np.matmul(np.transpose(eigenspace[i]), phi)
        sigma.append(w)
    x = 0
    minimum = float('inf')
    min_index = -1
    min_name = ""
    for train_image in all_images:
        similarity = np.linalg.norm(np.subtract(sigma, projections[x]))
        if similarity < minimum:
            minimum = similarity
            min_index = x
            min_name = train_image[9:]
        x += 1
    sample += 1
    if name[:4] == min_name[:4]:
        accuracy += 1
    test_image_out = img.imread(test_image)
    train_image_out = img.imread(all_images[min_index])
    imgs.append((test_image_out, train_image_out))

accuracy /= len(test_images)
print((k, accuracy))
plot.append((k, accuracy))

print(plot)
plt.scatter(*zip(*plot))
plt.show()
#fig = plt.figure(figsize=(12, 8))
columns = 12
rows = 8
x = 0
#for i in range(0, len(imgs)):
    #img1 = imgs[i][0]
    #img2 = imgs[i][1]
    #fig.add_subplot(rows, columns, x + 1)
    #plt.imshow(img1)
    #plt.axis('off')
    #fig.add_subplot(rows, columns, x + 2)
    #plt.imshow(img2)
    #plt.axis('off')
    #x += 2
#plt.show()
# In[ ]:




