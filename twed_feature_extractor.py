import numpy as np
import pandas as pd
import math
import datetime
import matplotlib.pyplot as plt
import os
import shutil


num_templates = 4 # 5 genuine samples of each user
num_remaining = 25 - num_templates
num_training_samples = 80 #70 users in training
batch_size = 2*num_remaining + num_templates # remaining 20 genuine and 20 forgery samples of one user
look_back = 1200 #length of the sequence
win_length = 3 # length of the capturing window in the feature extractor


feat_ext = '_clt_indp_400_len'

def linear_interp(seq,look_back):
	[len_x,len_y] = seq.shape
	num_times = int(look_back/len_x)
	new_len_x = num_times * len_x
	
	if(look_back <= len_x):
		print(".......No need of Interpolation......")
		return seq
	#print("........Duplicating.......")

	col_diffs = np.zeros((len_x,len_y)) # Assuming the 0th element is 0 Hence the diff between the 1st sample point from 0th point is 0
	col_diffs[0,:] = seq[0,:]/num_times

	for lv11 in range(1,len_x):
		col_diffs[lv11,:] = (seq[lv11,:] - seq[lv11-1,:])/num_times #caclulating the diffs

	new_seq = np.zeros((new_len_x,len_y))

	for lv12 in range(num_times):
		new_seq[lv12,:] = (lv12+1)*col_diffs[0,:] # interpolating the 1st element from 0 to the 1st element. Assuming that the 0th element is 0

	for lv10 in range(1,len_x):
		for lv11 in range(num_times):
			new_seq[lv10*num_times + lv11,:] = seq[lv10,:] + (lv11+1)*col_diffs[lv10,:]

	return new_seq


def averaging_filter(seq,winlen):
	print("***********Averaging***********")
	#print(seq.shape)
	[num_feat,len_seq] = seq.shape
	new_len = len_seq/winlen

	new_len = (int)(new_len)

	new_seq = np.zeros((num_feat,new_len))

	for lv10 in range(new_len):
		subseq = seq[:,lv10*winlen:lv10*winlen+winlen]
		new_seq[:,lv10] = np.transpose(subseq.sum(axis=1))
		new_seq[:,lv10] = new_seq[:,lv10]/((float)(winlen))
	#print(new_seq.shape)

	return new_seq



def preprocess(seq):
	[len_seq_x,len_seq_y] = seq.shape

	for lv9 in range(0,len_seq_y):
		for lv8 in range(0,len_seq_x-1):
			seq[lv8,lv9] = seq[lv8+1,lv9] - seq[lv8,lv9]

	seq = seq[:len_seq_x-1,:]
	#print(seq.shape)

	[len_seq_x,len_seq_y] = seq.shape
	seq_normalized = np.zeros((len_seq_x,len_seq_y))

	for lv6 in range(0,len_seq_y):
		mean_col_lv6 = np.mean(seq[:,lv6])
		std_col_lv6 = np.std(seq[:,lv6])

		for lv7 in range(0,len_seq_x):
			seq_normalized[lv7,lv6] = (seq[lv7,lv6]-mean_col_lv6)/std_col_lv6

	return seq_normalized



def twed_dist(seq_1,seq_2):
	[len_seq_1,feat_size] = seq_1.shape
	[len_seq_2,feat_size] = seq_2.shape
	twed_param_lambda = 0.1

	twed_mat = np.zeros((len_seq_1+1,len_seq_2+1))
	for lv4 in range(1,len_seq_1+1):
		twed_mat[lv4][0] = 9999999
	for lv4 in range(1,len_seq_2+1):
		twed_mat[0][lv4] = 9999999

	for lv4 in range(1,len_seq_1+1):
		for lv5 in range(1,len_seq_2+1):
			seq1_curr = seq_1[lv4-1,:]
			seq2_curr = seq_2[lv5-1,:]
			seq1_prev = seq_1[lv4-2,:]
			seq2_prev = seq_2[lv5-2,:]

			twed_val1 = twed_mat[lv4-1,lv5] + twed_param_lambda*np.linalg.norm(seq1_curr-seq1_prev)
			twed_val2 = twed_mat[lv4-1,lv5-1] + twed_param_lambda*np.linalg.norm(seq1_curr-seq2_curr) + twed_param_lambda*np.linalg.norm(seq1_prev-seq2_prev)
			twed_val3 = twed_mat[lv4,lv5-1] + twed_param_lambda*np.linalg.norm(seq2_prev-seq2_curr)

			twed_mat[lv4,lv5] = min(twed_val1,twed_val2)
			twed_mat[lv4,lv5] = min(twed_mat[lv4,lv5],twed_val3)


	score = twed_mat[len_seq_1][len_seq_2] #not normalized TWED score

	len_warping_path = 1

	wp_mat = np.zeros((len_seq_1+1,len_seq_2+1))


	x_axis = len_seq_1
	y_axis = len_seq_2

	wp_mat[len_seq_1-x_axis][len_seq_2-y_axis] = 1


	while(x_axis!=0 or y_axis!=0):
		if(x_axis==0):
			len_warping_path = len_warping_path + 1
			y_axis = y_axis - 1
			wp_mat[len_seq_1-x_axis][len_seq_2-y_axis] = len_warping_path
			continue
		elif(y_axis==0):
			len_warping_path = len_warping_path + 1
			x_axis = x_axis - 1
			wp_mat[len_seq_1-x_axis][len_seq_2-y_axis] = len_warping_path
			continue
		else:
			len_warping_path = len_warping_path + 1
			twed_val1 = twed_mat[x_axis-1][y_axis]
			twed_val2 = twed_mat[x_axis-1][y_axis-1]
			twed_val3 = twed_mat[x_axis][y_axis-1]
			mini = min(twed_val1,twed_val2)
			mini = min(mini,twed_val3)
			if(mini==twed_val1):
				x_axis = x_axis - 1
				continue
			elif(mini==twed_val2):
				y_axis = y_axis - 1
				continue
			else:
				x_axis = x_axis - 1
				y_axis = y_axis - 1

			wp_mat[len_seq_1-x_axis][len_seq_2-y_axis] = len_warping_path

	normalized_score = score/float(len_warping_path)

	return twed_mat/float(len_warping_path)
	


path = 'C:/Users/Sarojit Auddya/mcyt/mcyt/'
mod_path = 'D:/mcyt_400_len/'
ext = '_mod'
fname_pref = '00'
winlen = 3

for lv1 in range(0,100):
	num = ''
	if(lv1<10):
		num = '0' + str(lv1)
	else:
		num = str(lv1)

	new_folder_name = fname_pref + num + feat_ext
	new_folder_path = mod_path + fname_pref + num + feat_ext + '/'
	if(os.path.exists(new_folder_path)):
		shutil.rmtree(new_folder_path)


	os.mkdir(new_folder_path)



	folder_path = path + fname_pref + num + ext + '/'
	forgery_sig_pref = fname_pref + num + 'f' #prefix of files of forgery signature data of each user
	genuine_sig_pref = fname_pref + num + 'g' #same as above for genuine signature data

	for lv3 in range(0,num_remaining):
		dumping_fname_genuine = new_folder_path + 'g' + str(lv3) + '.csv'
		dumping_fname_forgery = new_folder_path + 'f' + str(lv3) + '.csv'

		filename_gen_lv3 = folder_path + genuine_sig_pref + str(num_templates+25+lv3) + '.csv'
		user_lv1_gen_sig_lv3 = pd.read_csv(filename_gen_lv3,sep=',',header=None)

		filename_for_lv3 = folder_path + forgery_sig_pref + str(num_templates+lv3) + '.csv'
		user_lv1_for_sig_lv3 = pd.read_csv(filename_for_lv3,sep=',',header=None)

		user_lv1_gen_sig_lv3_np = user_lv1_gen_sig_lv3.to_numpy()
		user_lv1_gen_sig_lv3_np	= user_lv1_gen_sig_lv3_np[:,:5]
		user_lv1_for_sig_lv3_np = user_lv1_for_sig_lv3.to_numpy()
		user_lv1_for_sig_lv3_np = user_lv1_for_sig_lv3_np[:,:5]

		user_lv1_gen_sig_lv3_np = preprocess(user_lv1_gen_sig_lv3_np)
		user_lv1_for_sig_lv3_np = preprocess(user_lv1_for_sig_lv3_np)


		gen_feat_ext = np.zeros((num_templates,look_back))
		for_feat_ext = np.zeros((num_templates,look_back))

		for lv2 in range(0,num_templates):		
			filename_template_lv1 = folder_path + genuine_sig_pref +  str(25+lv2) + '.csv'
			user_lv1_template_lv2 = pd.read_csv(filename_template_lv1,sep=',',header=None)
			user_lv1_template_lv2_np = user_lv1_template_lv2.to_numpy()
			user_lv1_template_lv2_np = user_lv1_template_lv2_np[:,:5]

			user_lv1_template_lv2_np = preprocess(user_lv1_template_lv2_np)

			twed_mat_tmp_gen = twed_dist(user_lv1_template_lv2_np,user_lv1_gen_sig_lv3_np)
			twed_mat_tmp_for = twed_dist(user_lv1_template_lv2_np,user_lv1_for_sig_lv3_np)

			[twed_mat_x,twed_mat_y] = twed_mat_tmp_gen.shape
			gen_feat = twed_mat_tmp_gen[twed_mat_x-1,2:]
			#print(gen_feat.shape)
			gen_feat = gen_feat.reshape(gen_feat.shape[0],1)

			#print(".........Genuine sample Before Interpolation.........")
			#print(gen_feat.shape)
			gen_feat = linear_interp(gen_feat,look_back)
			#print(".........Geuine sample After Interpoaltion.........")
			#print(gen_feat.shape)


			[twed_mat_x,twed_mat_y] = twed_mat_tmp_for.shape
			for_feat = twed_mat_tmp_for[twed_mat_x-1,2:]
			for_feat = for_feat.reshape(for_feat.shape[0],1)
			
			#print(".........Forgery sample Before Interpolation.........")
			#print(for_feat.shape)
			for_feat = linear_interp(for_feat,look_back)
			#print(".........Forgery sample After Interpolation.........")
			#print(for_feat.shape)


			gen_feat = gen_feat.reshape(1,gen_feat.shape[0])
			for_feat = for_feat.reshape(1,for_feat.shape[0])


			[dummy,len_gen_feat] = gen_feat.shape
			num_zeros_to_add = abs(look_back - len_gen_feat)
			zero_padding = np.zeros((1,num_zeros_to_add))
			gen_feat_padded = np.append(zero_padding,gen_feat,axis=1)




			[dummy,len_for_feat] = for_feat.shape
			num_zeros_to_add = look_back - len_for_feat
			for_feat_padded = []
			if(num_zeros_to_add>0):
				zero_padding = np.zeros((1,num_zeros_to_add))
				for_feat_padded = np.append(zero_padding,for_feat,axis=1)
			elif(num_zeros_to_add<0):
				for_feat_padded = for_feat[0,0:look_back]
			else:
				for_feat_padded = for_feat

			gen_feat_ext[lv2,:] = gen_feat_padded
			for_feat_ext[lv2,:] = for_feat_padded


		#gen_feat_filtered = gen_feat_ext
		#for_feat_filtered = for_feat_ext

		gen_feat_filtered = averaging_filter(gen_feat_ext,winlen) #window length 3
		for_feat_filtered = averaging_filter(for_feat_ext,winlen)

		print("******************************* User: "+ str(lv1) + " Sample Number: "+str(lv3)+"*******************************")
		print(gen_feat_filtered.shape)
		print(for_feat_filtered.shape)

		np.savetxt(dumping_fname_genuine,gen_feat_filtered,delimiter=',')
		np.savetxt(dumping_fname_forgery,for_feat_filtered,delimiter=',')

	for lv3 in range(0,num_templates):
		dumping_fname_forgery = new_folder_path + 'f' + str(num_remaining + lv3) + '.csv'

		filename_for_lv3 = folder_path + forgery_sig_pref + str(lv3) + '.csv'
		user_lv1_for_sig_lv3 = pd.read_csv(filename_for_lv3,sep=',',header=None)

		user_lv1_for_sig_lv3_np = user_lv1_for_sig_lv3.to_numpy()
		user_lv1_for_sig_lv3_np = user_lv1_for_sig_lv3_np[:,:5]

		user_lv1_for_sig_lv3_np = preprocess(user_lv1_for_sig_lv3_np)

		for_feat_ext = np.zeros((num_templates,look_back))

		for lv2 in range(0,num_templates):		
			filename_template_lv1 = folder_path + genuine_sig_pref +  str(25+lv2) + '.csv'
			user_lv1_template_lv2 = pd.read_csv(filename_template_lv1,sep=',',header=None)
			user_lv1_template_lv2_np = user_lv1_template_lv2.to_numpy()
			user_lv1_template_lv2_np = user_lv1_template_lv2_np[:,:5]

			user_lv1_template_lv2_np = preprocess(user_lv1_template_lv2_np)

			twed_mat_tmp_for = twed_dist(user_lv1_template_lv2_np,user_lv1_for_sig_lv3_np)

			[twed_mat_x,twed_mat_y] = twed_mat_tmp_for.shape
			for_feat = twed_mat_tmp_for[twed_mat_x-1,2:]
			for_feat = for_feat.reshape(for_feat.shape[0],1)
			#print(".........Forgery sample Before Interpolation.........")
			#print(for_feat.shape)
			for_feat = linear_interp(for_feat,look_back)
			#print(".........Forgery sample After Interpolation.........")
			#print(for_feat.shape)

			for_feat = for_feat.reshape(1,for_feat.shape[0])


			[dummy,len_for_feat] = for_feat.shape
			#print(for_feat.shape)
			num_zeros_to_add = look_back - len_for_feat
			for_feat_padded = []
			if(num_zeros_to_add>0):
				zero_padding = np.zeros((1,num_zeros_to_add))
				for_feat_padded = np.append(zero_padding,for_feat,axis=1)
			elif(num_zeros_to_add<0):
				for_feat_padded = for_feat[0,0:look_back]
			else:
				for_feat_padded = for_feat



			for_feat_ext[lv2,:] = for_feat_padded

		#for_feat_filtered = for_feat_ext
		for_feat_filtered = averaging_filter(for_feat_ext,winlen) #taking window length 3

		print("******************************* User: "+ str(lv1) + " Sample Number: "+str(num_remaining + lv3)+"*******************************")
		print(for_feat_filtered.shape)

		np.savetxt(dumping_fname_forgery,for_feat_filtered,delimiter=',')























