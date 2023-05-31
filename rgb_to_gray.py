from PIL import Image
import os

# 将图像转为灰度图
path = '/home/panding/code/UR/UR/demo4'
file_list = os.listdir(path)
for file in file_list:
    I = Image.open(path+"/"+file)
    L = I.convert('L')
    L.save(path+"/"+file)
    #print(file)