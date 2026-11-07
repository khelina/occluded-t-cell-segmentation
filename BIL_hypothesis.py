import numpy as np
import cv2
import os
import random


im_size=208
cell_radius=20
step_size =int(cell_radius/2)
#############################
from_ =cell_radius+step_size
to_ =im_size-from_
#################################

#####################################

###################################
def create_image(im_size,from_,to_,cell_radius, folder, curr_name, prev_name,label_name):
 prev_image_1=np.zeros((im_size,im_size), np.uint8)
 #prev_image_2=np.zeros((im_size,im_size), np.uint8)
 curr_image_1=np.zeros((im_size,im_size), np.uint8)
 #curr_image_2=np.zeros((im_size,im_size), np.uint8)
 label_image=np.zeros((im_size,im_size), np.uint8)
  #############################################
 x1_prev,y1_prev = random.randint(from_, to_),random.randint(from_, to_)
 cv2.circle(prev_image_1, (x1_prev, y1_prev), cell_radius, 1, -1)
 while True:
   prev_image_2=np.zeros((im_size,im_size), np.uint8)
   x2_prev,y2_prev = random.randint(from_, to_),random.randint(from_, to_)
   cv2.circle(prev_image_2, (x2_prev, y2_prev), cell_radius, 2, -1)
   prev_image=prev_image_1+prev_image_2
   if not np.any(prev_image == 3):
       cv2.imwrite(os.path.join(folder,prev_name), prev_image)
       break
   else:
       continue
##############################
 shift=int(cell_radius/2)
 x1_curr,y1_curr = random.randint(x1_prev-shift, x1_prev+shift),random.randint(y1_prev-shift, y1_prev+shift)
 cv2.circle(curr_image_1, (x1_curr, y1_curr), cell_radius, 100, -1)
 while True:
   curr_image_2=np.zeros((im_size,im_size), np.uint8)
   x2_curr,y2_curr = random.randint(x2_prev-shift, x2_prev+shift),random.randint(y2_prev-shift, y2_prev+shift)
   cv2.circle(curr_image_2, (x2_curr, y2_curr), cell_radius, 100, -1)
   curr_image=curr_image_1+curr_image_2
   if not np.any(curr_image == 200):
       cv2.circle(label_image, (x1_curr, y1_curr), cell_radius, 1, -1)
       cv2.circle(label_image, (x2_curr, y2_curr), cell_radius, 2, -1)
       cv2.imwrite(os.path.join(folder,curr_name), curr_image)
       cv2.imwrite(os.path.join(folder,label_name), label_image)
       break
   else:
       continue
   
####################################
folder=r"C:\Users\helina\Desktop\BIL\BIL_TRAIN_SET"
curr_name="curr.tif"
prev_name="prev.tif"
label_name="label.tif"
for k in range(200):
    curr_name ="curr_%s" % k +".tif"
    prev_name ="prev_%s" % k +".tif"
    label_name ="label_%s" % k +".tif"
    create_image(im_size,from_,to_,cell_radius, folder, curr_name, prev_name,label_name)
      
    
    

 




