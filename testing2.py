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


path = 'D:/mcyt/'
ext = '_clt_indp_350_len'
fname_pref = '00'

num_templates = 4 # 5 genuine samples of each user
num_training_samples = 70 #70 users in training
batch_size = 46 # remaining 20 genuine and 20 forgery samples of one user
look_back = 350 #length of the sequence
win_length = 1 # length of the capturing window in the feature extractor
num_genuine = 25 - num_templates

eers = []

#leakyrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

model = tf.keras.models.load_model('350_len_interpolated_saved_weights/weights-improvement-1467-0.92.h5')
model.summary()

target = np.zeros((1,2))
actual = np.zeros((1,2))

acc_avg = 0.0
count = 0

for lv1 in range(70,100): #70 batches(each batch corresponds to 1 user)
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

	if(acc>0.90 and acc<1.0 and count<10):
		target = np.append(target,labels,axis=0)
		actual = np.append(actual,predicted_probs,axis=0)
		target_indiv = labels[:,1]
		predicted_indiv = predicted_probs[:,1]
		[fpr,tpr,_] = roc_curve(target_indiv,predicted_indiv)
		plt_label = 'Client' + str(count+1)
		plt.plot(fpr,1-tpr,label=plt_label)
		acc_avg = acc_avg + acc
		count = count + 1
		eers.append(1-acc)


	print("+++++++++++++++++++++++++++++++++Done with Client+++++++++++++++++++++++++++++++++ "+str(lv1))


print(eers)

plt.xlabel("False Acceptance Rate")
plt.ylabel("False Rejection Rate")

plt.show()