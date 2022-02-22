# import numpy
import numpy as np

# Create and print a 3 by 3 array where every number is a 15
arr_15 = np.ones((3,3)) * 15
print(arr_15)
# print out what are the largest and smalled values in the array below
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr.max(),"\n",arr.min())
# import pyplot lib from matplotlib and Image lib from PIL
import matplotlib.pyplot as plt
from PIL import Image
# use PIL and matplotlib to read and display the ../data/zebra.jpg image
image = Image.open('./data/zebra.jpg')
print(image)
plt.imshow(image)
# convert the image to a numpy arrary and print the shape of the arrary
image_arr = np.array(image)
print(image_arr.shape)
# use slicing to set the RED and GREEN channels of the picture to 0, then use imshow() to show the isolated blue channel
image_arr_blue = image_arr.copy()
image_arr_blue[:,:,0] = 0
image_arr_blue[:,:,1] = 0
plt.imshow(image_arr_blue)

