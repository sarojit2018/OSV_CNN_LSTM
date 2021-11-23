import tensorflow as tf
import numpy as np
import pandas as pd
import math
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense,LSTM,Input,Dropout,GRU,Masking,Conv1D,AveragePooling1D,Reshape,Flatten,MaxPooling1D,BatchNormalization
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.optimizers import Adam,Adamax,SGD
from scipy import interpolate,stats
from scipy.spatial import distance

path = 'D:/mcyt_interp/'
ext = '_clt_indp_1200_len'
fname_pref = '00'

num_templates = 4 # 5 genuine samples of each user
num_training_samples = 90 #90 users in training
batch_size = 46 # remaining 20 genuine and 20 forgery samples of one user
look_back = 1200 #length of the sequence (after the windowing)
win_length = 1 # length of the capturing window in the feature extractor
num_genuine = 25 - num_templates

def scheduler(epoch):
	if(epoch < 1500):
		return 0.005
	elif(epoch in range(1500,3000)):
		return 0.005*0.999
	else:
		return 0.0011 





visible = Input(shape = (look_back,num_templates))
#x = Masking(mask_value = 0)(visible)
avg_pool = AveragePooling1D(pool_size=2)(visible) #reduce noise

conv_lay_1 = Conv1D(filters=12,kernel_size=3,activation='elu',padding='same')(avg_pool)
max_pool_1 = MaxPooling1D(pool_size=2)(conv_lay_1)
#max_pool_1 = BatchNormalization()(max_pool_1)

conv_lay_2 = Conv1D(filters=12,kernel_size=3,activation='elu',padding='same')(max_pool_1)
max_pool_2 = MaxPooling1D(pool_size=2)(conv_lay_2)
#max_pool_2 = BatchNormalization()(max_pool_2)

conv_lay_3 = Conv1D(filters=12,kernel_size=3,activation='elu',padding='same')(max_pool_2)
max_pool_3 = MaxPooling1D(pool_size=2)(conv_lay_3)
#max_pool_3 = BatchNormalization()(max_pool_3)

conv_lay_4 = Conv1D(filters=12,kernel_size=3,activation='elu',padding='same')(max_pool_3)
max_pool_4 = MaxPooling1D(pool_size=2)(conv_lay_4)
#max_pool_4 = BatchNormalization()(max_pool_4)



lstm_lay_1 = LSTM(10,activation='linear',return_sequences=True)(max_pool_4)
lstm_lay_2 = LSTM(8,activation='linear',return_sequences=False)(lstm_lay_1)
#conv_lay_1 = Conv1D(filters=16,kernel_size=3,activation='relu',padding='same')(lstm_lay_1)
#avg_pool_1 =  AveragePooling1D(pool_size=2)(conv_lay_1)
#flat = Flatten()(lstm_lay_2)

#lstm_lay_2 = LSTM(2,activation='softmax',return_sequences=False)(lstm_lay_1)

output = Dense(2, activation='softmax')(lstm_lay_2)

#conv_lay_2 = Conv1D(filters=8,kernel_size=3,activation='relu',padding='same')(avg_pool_1)
#avg_pool_2 = AveragePooling1D(pool_size=2)(conv_lay_2)

#lstm_lay_3 = LSTM(2,activation='softmax',return_sequences=False)(lstm_lay_2)


#flat_lay_1 = Flatten()(avg_pool_5)
#dense_lay_1 = Dense(10,activation='relu')(flat_lay_1)

#dense_lay_1 = Dense(10,activation=leakyrelu)(flat_lay_1)

filepath="1200_len_interpolated_saved_weights/weights-improvement-{epoch:04d}-{binary_accuracy:.2f}-{val_binary_accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=0, mode='auto',min_delta=0.0001, cooldown=0, min_lr=0.0)
#reduce_LR = tf.keras.callbacks.LearningRateScheduler(scheduler)
callbacks_list = [checkpoint,reduce_LR]



model = Model(inputs=visible,outputs=output)

optimizer = Adamax(lr=0.0045)
#optimizer = SGD(lr=0.1) 

model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['binary_accuracy'])

model.summary()



def data_gen(): #data generator
	while(1):
		for lv1 in range(0,90): #90 batches(each batch corresponds to 1 user)
			num = ''
			if(lv1<10):
				num = '0' + str(lv1)
			else:
				num = str(lv1)
			folder_path = path + fname_pref + num + ext + '/'
			forgery_sig_pref = folder_path + 'f' #prefix of files of forgery signature data of each user
			genuine_sig_pref = folder_path + 'g' #same as above for genuine signature data 

			feat_mat_x = np.zeros((batch_size,look_back,num_templates))
			labels = np.zeros((batch_size,2))


			for lv2 in range(num_genuine):
				fname_for = forgery_sig_pref + str(lv2) + '.csv'
				fname_gen = genuine_sig_pref + str(lv2) + '.csv'

				gen_feat = pd.read_csv(fname_for,sep=',',header=None)
				for_feat = pd.read_csv(fname_gen,sep=',',header=None)

				for_feat = for_feat.to_numpy()
				for_feat = for_feat.astype(np.float32)

				gen_feat = gen_feat.to_numpy()
				gen_feat = gen_feat.astype(np.float32)

				feat_mat_x[lv2,:,:] = gen_feat.T
				labels[lv2,0] = 1

				feat_mat_x[lv2 + num_genuine,:,:] = for_feat.T
				labels[lv2 + num_genuine,1] = 1

			for lv4 in range(num_templates):
				fname_for = forgery_sig_pref + str(num_templates + lv4) + '.csv'
				for_feat = pd.read_csv(fname_gen,sep=',',header=None)

				for_feat = for_feat.to_numpy()
				for_feat = for_feat.astype(np.float32)

				feat_mat_x[lv4 + 2*num_genuine,:,:] = for_feat.T
				labels[lv4 + 2*num_genuine,1] = 1

			yield(feat_mat_x,labels)


def valid_gen():
	while(1):
		validation_data = np.zeros((46,look_back,num_templates))
		validation_labels = np.zeros((46,2))
		#validation_labels = []
		for lv1 in range(90,100): #90 batches(each batch corresponds to 1 user)
			num = ''
			if(lv1<10):
				num = '0' + str(lv1)
			else:
				num = str(lv1)
			folder_path = path + fname_pref + num + ext + '/'
			forgery_sig_pref = folder_path + 'f' #prefix of files of forgery signature data of each user
			genuine_sig_pref = folder_path + 'g' #same as above for genuine signature data 

			feat_mat_x = np.zeros((batch_size,look_back,num_templates))
			labels = np.zeros((batch_size,2))


			for lv2 in range(num_genuine):
				fname_for = forgery_sig_pref + str(lv2) + '.csv'
				fname_gen = genuine_sig_pref + str(lv2) + '.csv'

				gen_feat = pd.read_csv(fname_for,sep=',',header=None)
				for_feat = pd.read_csv(fname_gen,sep=',',header=None)

				for_feat = for_feat.to_numpy()
				for_feat = for_feat.astype(np.float32)

				gen_feat = gen_feat.to_numpy()
				gen_feat = gen_feat.astype(np.float32)

				feat_mat_x[lv2,:,:] = gen_feat.T
				labels[lv2,0] = 1

				feat_mat_x[lv2 + num_genuine,:,:] = for_feat.T
				labels[lv2 + num_genuine,1] = 1

			for lv4 in range(num_templates):
				fname_for = forgery_sig_pref + str(num_templates + lv4) + '.csv'
				for_feat = pd.read_csv(fname_gen,sep=',',header=None)

				for_feat = for_feat.to_numpy()
				for_feat = for_feat.astype(np.float32)

				feat_mat_x[lv4 + 2*num_genuine,:,:] = for_feat.T
				labels[lv4 + 2*num_genuine,1] = 1

			#validation_data[lv1*batch_size:(lv1+1)*batch_size,:,:] = feat_mat_x[:,:,:]
			#validation_labels[lv1*batch_size:(lv1+1)*batch_size,:] = labels[:,:]

			validation_data = np.append(validation_data,feat_mat_x,axis=0)
			validation_labels = np.append(validation_labels,labels,axis=0)

		validation_data = validation_data[46:,:,:]
		validation_labels = validation_labels[46:,:]

		return(validation_data,validation_labels)


#modelcheckpoint = ModelCheckpoint()

model.fit_generator(data_gen(),steps_per_epoch=80,epochs=5000,verbose=1,validation_data=valid_gen(),validation_steps=1,callbacks=callbacks_list)

model.save('clt_indp_classifier.h5')