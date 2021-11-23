import tensorflow as tf
import numpy as np
import pandas as pd
import math
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,LSTM,Input,Dropout,GRU,Masking,Conv1D,AveragePooling1D,Reshape,Flatten
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam,Adamax
from scipy import interpolate,stats
from scipy.spatial import distance
from sklearn.metrics import roc_curve,roc_auc_score

path = 'D:/mcyt_interp/'
ext = '_clt_indp_1200_len'
fname_pref = '00'

num_templates = 4 # 5 genuine samples of each user
num_training_samples = 70 #70 users in training
batch_size = 46 # remaining 20 genuine and 20 forgery samples of one user
look_back = 1200 #length of the sequence
win_length = 1 # length of the capturing window in the feature extractor
num_genuine = 25 - num_templates

#leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

model = tf.keras.models.load_model('1200_len_interpolated_saved_weights/weights/weights-improvement-0001-0.94.h5')
model.summary()

target = np.zeros((1,2))
actual = np.zeros((1,2))

acc_avg = 0.0
count = 0

for lv1 in range(90,100): #70 batches(each batch corresponds to 1 user)
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
	
	_,acc = model.evaluate(feat_mat_x,labels)
	predicted_probs = model.predict(feat_mat_x)

	#labels_T = labels.T
	#predicted_probs_T = predicted_probs.T

	if(acc>0):
		target = np.append(target,labels,axis=0)
		actual = np.append(actual,predicted_probs,axis=0)
		count = count + 1


	acc_avg = acc_avg + acc
	print("+++++++++++++++++++++++++++++++++Done with Client+++++++++++++++++++++++++++++++++ "+str(lv1))




print(acc_avg/10)

target0 = target[1:,0]
actual0 = actual[1:,0]

[fpr0,tpr0,_] = roc_curve(target0,actual0)

target1 = target[1:,1]
actual1 = actual[1:,1]

[fpr1,tpr1,_] = roc_curve(target1,actual1)

plt.plot(fpr1,1-tpr0,c='g',linewidth=2.0)



plt.show()
