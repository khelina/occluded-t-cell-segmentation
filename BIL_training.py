import tensorflow as tf
import keras
from keras.models import Model,load_model
from keras.layers import Input, concatenate,UpSampling2D
from keras.layers import  Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import LearningRateScheduler
#from keras.utils import multi_gpu_model
from keras.models import model_from_json
import pickle
import numpy as np
import os
from random import shuffle
import json
#import cv2
from PIL import Image, ImageDraw
###########################################################
Size=208.0
#Batch=16
Batch=1
N_channels=2
NUM_EPOCHS=1
INIT_LR=0.0009
################################################
def characters(x):# this is for sorting by cell number   "90001"
    return(x[-9:-4])
##########################################################################    
def load_and_organize_images(folder_dir,DATA):    
    names_current=[]
    names_previous=[]
    names_labels=[]          
    for filename in os.listdir(folder_dir):
           if "curr" in filename:
             curr_name=os.path.join(folder_dir, filename)
             #print("full_name=",full_name)             
             names_current.append(curr_name)       
           if "prev" in filename:
             prev_name=os.path.join(folder_dir, filename)        
             names_previous.append(prev_name)         
           if "label" in filename:
             label_name=os.path.join(folder_dir, filename)       
             names_labels.append(label_name)                    
    names_current_sorted=sorted(names_current,key=characters)
    #print("len(names_bright_sorted)=",len(names_bright_sorted))
    names_previous_sorted=sorted(names_previous,key=characters)
    #print("len(names_fluor_sorted)=",len(names_fluor_sorted))
    names_labels_sorted=sorted(names_labels,key=characters)
    #print("len(names_segmented_sorted)=",len(names_segmented_sorted))    
    n=len(names_labels_sorted)
    for i in range(n):        
        current=Image.open(names_current_sorted[i])       
        previous=Image.open(names_previous_sorted[i])             
        label=Image.open(names_labels_sorted[i])

        current=np.array(current)
        previous=np.array(previous)                   
        label=np.array(label)
        DATA.append((current,previous,label))
    return DATA
#########################################################
def pre_augment_train(data):
    for i in range(len(data)):
       new_curr=np.array(data[i][0])
       old_prev=np.array(data[i][1])
       old_label=np.array(data[i][2])
       ####################################       
       new_prev=np.zeros((208,208), np.uint8)       
       new_prev[old_prev==50]=100
       new_prev[old_prev==100]=50
       #####################################
       new_label=np.zeros((208,208), np.uint8)
       new_label[old_label==50]=100
       new_label[old_label==100]=50
       ##################################
       #new_curr = Image.fromarray(new_curr)
       #new_prev = Image.fromarray(new_prev)
       #new_label = Image.fromarray(new_label)
       ######################################
       data.append((new_curr,new_prev, new_label))
    return data
##################################
"""
def check(data):
    check_folder=r"C:\Users\helina\Desktop\BIL\CHECK"
    for i in range(len(data)):
        curr=data[i][0]
        prev=data[i][1]
        label=data[i][2]
        curr.save(os.path.join(check_folder,"curr_%s.tiff" % i))
        prev.save(os.path.join(check_folder,"prev_%s.tiff" % i))
        label.save(os.path.join(check_folder,"label_%s.tiff" % i))
#####################################################        
folder_dir=r"C:\Users\helina\Desktop\BIL\BIL_TRAIN_SET"
DATA=[]       
data=load_and_organize_images(folder_dir,DATA)
data=pre_augment_train(data)
check(data)
"""
##############################################################    
def augment_train(data):
 print("inside augment_train")
 methods=[Image.Transpose.FLIP_LEFT_RIGHT,Image.Transpose.FLIP_TOP_BOTTOM,Image.Transpose.ROTATE_90,Image.Transpose.ROTATE_180,Image.Transpose.ROTATE_270]
 for i in range(len(data)):
    init_sample=data[i]
    #np_sample=[]
    
    for method in methods:          
       new_sample=[]
       for k in range(5):        
         init_copy=init_sample[k].copy()   
         #im11 = cv2.flip( im1, 1)
         transposed=init_copy.transpose(method)
         transposed_array=np.array(transposed)
         #print("transposed_array.shape=",transposed_array.shape)
         new_sample.append(transposed_array)
       data.append(new_sample)
    
    np_sample=[np.array(init_sample[p]) for p in range(5)]
    data[i]=np_sample            
 return data

###########################################################################################
def create_samples_for_BIL(input_data, data_type):# data_type is either TRAIN or VALID 
    if data_type=="TRAIN":
        input_data=pre_augment_train(input_data)         
        #input_data=augment_train(input_data)
        shuffle(input_data)
    a=(len(input_data)//Batch)*Batch
    input_data=input_data[:a]
    N_data=len(input_data)
    data_inputs=np.zeros((N_data,int(Size),int(Size),N_channels))
    data_labels=np.zeros((N_data,int(Size),int(Size)))
    #data_weights=np.zeros((N_data,int(Size),int(Size)))
    
    for pp in range(N_data):
     frame=np.zeros((int(Size),int(Size),N_channels))
     image=input_data[pp][0]
     #print("INSIDE create_samples_for_segmentor")
     #print("image.shape=", image.shape)
     #print("image.dtype=", image.dtype)
     frame[:,:,0]=input_data[pp][0].astype(np.float32)
     frame[:,:,1]=input_data[pp][1].astype(np.float32)
     #frame[:,:,2]=input_data[pp][2].astype(np.float32)    
    
     means = frame.mean(axis=(0,1), dtype='float64')
     stds = frame.std(axis=(0,1), dtype='float64')    
     frame = (frame - means) / stds    
     data_inputs[pp,:,:,:]=frame
     #######################################
     label0=input_data[pp][2].astype(np.float32)
     labell=label0/255.0
     data_labels[pp,:,:]=labell
     
     #weight=input_data[pp][3].astype(np.float32)     
     #data_weights[pp,:,:]=weight      

    data_labels=data_labels.reshape((N_data,int(Size),int(Size),1))
    #data_weights=data_weights.reshape((N_data,int(Size),int(Size),1)) 
    return data_inputs,data_labels 
##############################################################################################

##############################################################################################
my_dir = os.getcwd()
########### This is for supercomputer training on new movies
#main_folder=os.path.join(my_dir, "SEGMENTOR_TRAINING")
################### This is for my desltop

main_folder =os.path.join(my_dir,"Desktop")
#main_folder =my_dir

############################for debugging - training on old movies
#main_folder=my_dir

################################

print("main_folder=", main_folder)
#main_folder=r"C:\Users\helina\Desktop\SEGMENTOR_TRAINING"
train_directory=os.path.join(main_folder,"BIL","BIL_TRAIN_SET")
#train_directory=os.path.join(main_folder,"SEGMENTOR_TRAIN_SET")

print("train_directory=",train_directory)

TRAIN=[]
 
TRAIN=load_and_organize_images(train_directory,TRAIN)
print("TRAIN=", len(TRAIN))

train_inputs, train_labels =create_samples_for_BIL(TRAIN, "TRAIN")         
print("TRAIN after augmentation=", len(TRAIN))

##################################################################################################
valid_directory =os.path.join(main_folder,"BIL","BIL_VALID_SET")
#valid_directory=os.path.join(main_folder,"SEGMENTOR_VALID_SET")

VALID=[]
VALID=load_and_organize_images(valid_directory,VALID)
print("VALID=", len(VALID)) 
valid_inputs,   valid_labels =create_samples_for_BIL(VALID, "VALID")         
#validation_data=([valid_inputs,valid_weights],valid_labels) 
###########################################
#################################
def IOU(y_true, y_pred):
    n_classes = 3
    eps = tf.keras.backend.epsilon()
    true_cls = tf.cast(y_true[..., 0], tf.int64)   # (batch, H, W)
    pred_cls = tf.argmax(y_pred, axis=-1)          # (batch, H, W), int64
    ious = []
    for k in range(1, n_classes):                  # skip background
        t = tf.equal(true_cls, k)
        p = tf.equal(pred_cls, k)
        inter = tf.reduce_sum(tf.cast(tf.logical_and(t, p), tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(t, p), tf.float32))
        ious.append((inter + eps) / (union + eps))
    return tf.add_n(ious) / len(ious)
####################################################

##############################################################
def get_compiled_model():
  input_image = Input(shape=(208,208,N_channels))
  #weights_tensor = Input(shape=(int(Size),int(Size),1))
  conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_image)
  conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
  conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
  #drop2 = Dropout(0.3)(conv2)# added myself
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
  conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
  #drop3 = Dropout(0.3)(conv3)# added myself
  pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

  conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
  conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
  drop4 = Dropout(0.3)(conv4)
  pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
  conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
  drop5 = Dropout(0.3)(conv5)

  up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
  merge6 = concatenate([drop4,up6], axis = 3)
  conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
  conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

  up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
  conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

  up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
  conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

  up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
  conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  #conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
  
  #y_pred  = Conv2D(1, 1, activation = 'sigmoid')(conv9)
  y_pred = Conv2D(3, 1, activation='softmax')(conv9)# 3 neurons corresponding to 0,1,2
  ################################################
  ##############################
  model = Model(inputs=input_image, outputs=y_pred)
 
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[IOU]) 
  
  return model
###########################################################

###############################################################################################
###############  This is a better training yielding  perfect learning curves. The idea is to evaluate training set after each epoch without weights ()
######################################################################################
List_of_train_errors=[]
def evaluate_train_and_save(model,train_inputs, train_labels):
   test_weights=np.ones((train_weights.shape))
   score = model.evaluate([train_inputs,test_weights], train_labels, verbose = 0)
   print("corrected train score=", score)
   List_of_train_errors.append(score)

List_of_valid_errors=[]
def evaluate_valid_and_save(model,valid_inputs,valid_weights,valid_labels):
   score = model.evaluate([valid_inputs,valid_weights], valid_labels, verbose = 0)
  
   print("valid score=", score)
   List_of_valid_errors.append(score)
   
def poly_decay(epoch):
 maxEpochs = NUM_EPOCHS
 baseLR = INIT_LR
 power = 2.0
 alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
 return alpha
###########################################
def scheduler(epoch, lr):
   return lr * tf.math.exp(-0.1)
####################################################################################
callbacks = [LearningRateScheduler(scheduler)]
#model.compile(optimizer = 'adam', loss = my_loss(weights_tensor), metrics = [IOU])
#model.save("SEGMENTOR.h5")


print("INIT_LR=", INIT_LR)
callbacks = [LearningRateScheduler(poly_decay)]
List_of_train_errors=[]
History_of_training=[]
learn_rate=INIT_LR

print("train_inputs.shape=",train_inputs.shape)
print("train_labels.shape=",train_labels.shape)
print("train_weights.shape=",train_weights.shape)
print("valid_inputs.shape=",valid_inputs.shape)
print("valid_labels.shape=",valid_labels.shape)
print("valid_weights.shape=",valid_weights.shape)
##########################
print("train_inputs.dtype=",train_inputs.dtype)
print("train_labels.dtype=",train_labels.dtype)
print("train_weights.dtype=",train_weights.dtype)
print("valid_inputs.dtype=",valid_inputs.dtype)
print("valid_labels.dtype=",valid_labels.dtype)
print("valid_weights.dtype=",valid_weights.dtype)
print("labels min/max/sum:", np.min(train_labels), np.max(train_labels), np.sum(train_labels))
print("weights min/max/sum:", np.min(train_weights), np.max(train_weights), np.sum(train_weights))
###################################
#print("train_inputs.shape=",train_inputs.shape)
#####################################
"""
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
with strategy.scope():    
    model = get_compiled_model()
"""
model = get_compiled_model()
#history=model.fit(train_samples, train_labels_mult, validation_data=validation_data, batch_size=batch, epochs=NUM_EPOCHS,callbacks=callbacks, shuffle=True, verbose=2)
#history=model.fit([train_inputs,train_weights], train_labels, validation_data=validation_data, batch_size=Batch, epochs=NUM_EPOCHS,callbacks=callbacks, shuffle=True, verbose=2)
#history = model.fit([train_inputs, train_weights], train_labels, validation_data=validation_data, batch_size=Batch, epochs=NUM_EPOCHS, callbacks=callbacks, shuffle=True, verbose=2)

###################################
print("checking for nuns in inputs")
for name, arr in (("X", train_inputs), ("y", train_labels)):
    print(name, np.isnan(arr).any(), np.isinf(arr).any())
print("BEFORE TRAINING")

preds = model.predict([train_inputs[:4], train_labels[:4]])
print("preds min/max/mean:", preds.min(), preds.max(), preds.mean())
print("frac >0.5:", np.mean(preds>0.5))
#################################
history = model.fit(
    x=train_inputs,
    y=train_labels,
    validation_data=(valid_inputs,  valid_labels),
    batch_size=Batch,
    epochs=NUM_EPOCHS, 
    callbacks=callbacks,
    shuffle=True,
    verbose=2
)
########################################
print("After TRAINING")
preds = model.predict([train_inputs[:4], train_weights[:4], train_labels[:4]])
print("preds min/max/mean:", preds.min(), preds.max(), preds.mean())
print("frac >0.5:", np.mean(preds>0.5))
"""
for epoch in range(NUM_EPOCHS):  
  print("REAL_EPOCH_NUMBER=", epoch+1, end =" ")
  print("out of",NUM_EPOCHS)
  #parallel_model = multi_gpu_model(model, gpus=4)
  #parallel_model=model
  #parallel_model.compile(optimizer = 'adam', loss = my_loss(weights_tensor), metrics = [IOU])
  ###############################################
  #strategy = tf.distribute.MirroredStrategy()
  #print("Number of devices: {}".format(strategy.num_replicas_in_sync))
  #with strategy.scope():    
    #model = get_compiled_model()
  #history=model.fit(train_samples, train_labels_mult, validation_data=validation_data, batch_size=batch, epochs=NUM_EPOCHS,callbacks=callbacks, shuffle=True, verbose=2)
  ############################################
  history=model.fit([train_inputs,train_weights], train_labels, validation_data=validation_data, batch_size=Batch, epochs=1,callbacks=callbacks, shuffle=True, verbose=2)
  History_of_training.append(history.history)
  print("History of training=", History_of_training )
  evaluate_train_and_save(model,train_inputs, train_labels) 
  evaluate_valid_and_save(model,valid_inputs,valid_weights,valid_labels)
  #model.save("SEGMENTOR.h5")
  #learn_rate=scheduler(epoch,learn_rate)
"""    
####################################################################################
output_folder=os.path.join(main_folder,"SEGMENTOR_TRAINING","OUTPUTS")
print("I am going to save history")
print("history.history=", history.history)
name_history=os.path.join(output_folder,"SEGMENTOR_HISTORY.pickle")
with open(name_history, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
    
print("saved history in file", name_history)
print("I am going to save weights")
name_weights=os.path.join(output_folder,"SEGMENTOR_WEIGHTS.h5")  
model.save_weights(name_weights)
print("saved weights in file", name_weights)


#model.compile(optimizer = 'adam', loss=my_loss(weights_tensor),metrics=[IOU])
model_json=model.to_json()
with open(os.path.join(output_folder,"SEGMENTOR_MODEL.json"),"w") as json_file:
   json_file.write(model_json)
print("saved the model in json")


print("I am going to save the whole model")
#model.save(os.path.join(output_folder,'SEGMENTOR_MODEL_WITH_WEIGHTS.h5'))
model.save(os.path.join(output_folder,'SEGMENTOR_MODEL_WITH_WEIGHTS.keras'))
print("saved the whole model") 

 
#model = load_model('SEGMENTOR.h5', custom_objects={ 'loss': my_loss(weights) })
######
#########################################################
################################### PLOT LEARNING CURVES #######################################################
"""
import matplotlib.pyplot as plt

n=len(List_of_train_errors)

train_loss=[List_of_train_errors[i][0] for i in range(n) ]
valid_loss=[List_of_valid_errors[i][0] for i in range(n) ]
plt.plot(valid_loss,"magenta")
plt.plot(train_loss,"green")
plt.title('')
plt.ylabel('mse loss')
plt.xlabel('epoch')
plt.legend(['validation loss ='+str(round(valid_loss[-1],5)), 'training loss = '+str(round(train_loss[-1],5))], loc='upper right')
#plt.show()
plt.savefig(os.path.join(output_folder,'loss.png'))
plt.close() 
train_IOU=[List_of_train_errors[i][1] for i in range(n) ]
valid_IOU=[List_of_valid_errors[i][1] for i in range(n) ]
plt.plot(valid_IOU,"red")
plt.plot(train_IOU,"blue")
plt.title('IOU')
plt.ylabel('IOU')
plt.xlabel('epoch')
plt.legend(['validation IOU ='+str(round(valid_IOU[-1],5)), 'training IOU = '+str(round(train_IOU[-1],5))], loc='lower right')
#plt.show()
plt.savefig(os.path.join(output_folder,'IoU.png'))




######################## plotting bad training curves ##############
x=History_of_training

val_loss=x


val_loss=[x[i]["val_loss"][0] for i in range(len(x))]
train_loss=[x[i]["loss"][0] for i in range(len(x))]

plt.plot(val_loss,"magenta")
plt.plot(train_loss,"green")
plt.title('')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['validation loss ='+str(round(val_loss[-1],5)), 'training loss = '+str(round(train_loss[-1],5))], loc='upper right')
#plt.show()

plt.savefig(os.path.join(output_folder,'loss.png'))
val_IOU=[x[i]["val_IOU"][0] for i in range(len(x))]
train_IOU=[x[i]["IOU"][0] for i in range(len(x))]


plt.plot(val_IOU,"red")
plt.plot(train_IOU,"blue")
plt.title('IOU')
plt.ylabel('IOU')
plt.xlabel('epoch')
plt.legend(['validation IOU ='+str(round(val_IOU[-1],5)), 'training IOU = '+str(round(train_IOU[-1],5))], loc='lower right')
#plt.show()

################################################################################################

################# TESTING ON OTHER MOVIES ###############
json_file_segmentor=open("SEGMENTOR.json","r")
model=json_file_segmentor.read()
json_file_segmentor.close()
model= model_from_json(model)
model.load_weights("SEGMENTOR.weights.h5")







test_directory="C:\\Users\\kfedorchuk\\SEGMENTATION FOR THESIS\\PATCHES FOR TESTING SEGMENTOR\\PATCHES from FIVE_Pos0701a"

fluorescent,brightfield,segmented,zero_frames,weights_maps=load_images(test_directory)              
print("images loaded")
print("number of test samples=", len(segmented))
if len(weights_maps)==0:
 print("no weights maps found")      
test_inputs, test_weights, test_labels=create_samples(fluorescent,brightfield,segmented,zero_frames,weights_maps, "VALID")
test_data=([test_inputs,test_weights],test_labels)

#model = load_model('SEGMENTOR.h5', custom_objects={ 'loss': my_loss(test_weights) })
#model = load_model('SEGMENTOR.h5')
model.compile(optimizer = 'adam', loss="binary_crossentropy",metrics=[IOU])

score = model.evaluate(test_inputs, test_labels, verbose = 0)
print("score=", score)
"""


