# OSV_CNN_LSTM
Online Signature Verification module using a Time Warp Edit Distance based Client Independent feature descriptor coupled with a CNN-LSTM based binary classifier. Dataset Used: MCYT-100. 

This project contains the following modules:

1. A Time Warp Edit Distance (TWED) based feature descriptor: The TWED feature descriptor is formulated to negate the client's characteristics from their signtatures to obtain a client independent feature sequence which is then used by a classifier to discriminate genuine and forgery.
2. We use a averaging window(of window length k) to reduce the feature sequence lengths before providing feature to classifier.
3. We use a Conv-LSTM based binary classifier to discriminate genuine and forgery samples. The 1-D Convolution modules constitutes the Local Feature Learning Blocks (LFLB) to capture short-sighted patterns in the feature sequence which is followed by LSTM based modules to capture longterm patterns in the output of the LFLBs. 
4. The Loss function used is Binary Crossentropy with equal weights to both classes.


# Pretrained Models: https://drive.google.com/drive/folders/1n_IjjLC47RRriJlosF6KbSNWidqcb_Ik?usp=sharing
