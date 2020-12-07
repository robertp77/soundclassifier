# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:50:33 2020

@author: arobu
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import plot_confusion_matrix, roc_curve, roc_auc_score
import pickle
import librosa
import librosa.display
from librosa.core import resample, to_mono
from glob import glob
import wavio
import os
from tqdm import tqdm
import argparse
from scipy.stats import mode
#PUT FILES TO BE CLASSIFIED IN validation WITHIN SAME FOLDER AS THIS CODE

# Class labels
instruments = ['Acoustic_guitar', 'Bass_drum', 'Cello', 'Clarinet', 
               'Double_bass', 'Flute', 'Hi_hat', 'Saxophone', 'Snare_drum', 'Violin']

# load the model from disk
f = open('Classifier_SVM_100_50_of_100_final.pckl', 'rb')
estimator = pickle.load(f)
f.close()


# Feature extraction step
# This is my own class. You can have yours here.

## Take absolute value of data and use rolling window to detect when maximum
##      signal dips below threshold
## If max > threshold, remove that data point (dont add it to mask)
def envelope(y, rate, threshold):
    mask = []   #list of all data points that we keep
    #y = pd.Series(y).apply(np.abs)  #absolute value of all data (becuase it is ocillating +/-)
    y = pd.Series(np.reshape(y, (len(y),))).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20), #creates rolling windows of data
                       min_periods=1,       #   and finds the mean of each
                       center=True).max()   #list of means of all windows
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean

def downsample_mono(path, sr):
    obj = wavio.read(path)
    wav = obj.data.astype(np.float32, order='F')
    rate = obj.rate
    try:
        tmp = wav.shape[1]
        wav = to_mono(wav.T)
    except:
        pass
    wav = wav.flatten()
    wav = resample(wav, rate, sr)
    wav = wav.astype(np.int16)
    #print(wav.shape)
    return sr, wav


def mfcc(args, sample):
    #src_root = args.src_root
    #wav_paths = glob('{}/**'.format(src_root), recursive=True)
    #wav_path = [x for x in wav_paths if fname in x]
    #if len(wav_path) != 1:
    #    print('audio file not found for sub-string: {}'.format(args.fn))
    #    return
    wav = sample
    #wav = obj.data.astype(np.float32, order='F')
    wav = wav.flatten()
    #rate = obj.rate
    rate=16000
    #wav = resample(wav, rate, args.sr)
    wav = wav.astype(np.float32)
    y=pd.Series(np.reshape(wav, (len(wav),))).apply(np.abs)
    y=y.to_numpy(dtype=np.float32)
    yy=librosa.feature.mfcc(y=y,sr=16000,S=None,n_mfcc=100,dct_type=2,norm='ortho',lifter=0)
    # Convert to deciBels
    #feats = np.mean(librosa.amplitude_to_db(np.abs(yy),ref = np.max),1)
    #feats = feats[:50]
    yy=yy[:50, :]
    feats=yy.ravel()
    feats = np.sum(yy, axis=1)
    return feats, yy


def classify_files(args):
    src_root = args.src_root
    dst_root = args.dst_root
    dt = args.delta_time

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_paths = [x for x in wav_paths if '.wav' in x]
    #check_dir(dst_root)
  
    classes = glob(('{}/*/'.format(src_root)))
    #print(classes)
    fnum=0 #File counter
    ytrue=[] #Holds true y vector
    files=[] #Holds file names
    y=[] #Holds predicted classes
   # MasterX=[]
    #Mastery=[]
  #  Masterytrue=[]
    for _cls in classes:
        #Take the folder name only. Needed bc glob gives the relative path.
        _cls = _cls.split('\\')[-2] 
        #print(_cls)
       # target_dir = os.path.join(dst_root, _cls)
        #check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
       # print(src_dir)
        iterfiles=glob('{}/*.wav'.format(src_dir))
       # print(iterfiles)
        for fpath in  tqdm(iterfiles): #tqdm(os.listdir(src_dir)):
            fn = fpath.split('\\')[-1]
            src_fn = os.path.join(src_dir, fn)
            #src_fn = fn
            #print(src_fn)
            rate, wav = downsample_mono(src_fn, args.sr)
            mask, y_mean = envelope(wav, rate, threshold=args.threshold)
            wav = wav[mask]
            delta_sample = int(dt*rate)
            samples=[] #Holds samples for this file
            # cleaned audio is less than a single sample
            # pad with zeros to delta_sample size
            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                sample[:wav.shape[0]] = wav
                samples.append(sample)
               # save_sample(sample, rate, target_dir, fn, 0)
            # step through audio and save every delta_sample
            # discard the ending audio if it is too short
            else:
                trunc = wav.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, wav.shape[0]-trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = wav[start:stop]
                    samples.append(sample)
                    #save_sample(sample, rate, target_dir, fn, cnt)
            X=np.ndarray((len(samples), 50)) #There are 50 mfcc features
            #print(samples)
            for samplenum, mysample in enumerate(samples):  
                feats, yy = mfcc(args, mysample)
                X[samplenum,:]=feats 
                #MasterX.append(feats)
               # Masterytrue.append(_cls)
            
                #print(X)
            #print(fpath)
            yvec=estimator.predict(X)
            #Mastery=np.concatenate((Mastery, yvec), axis=0)
            #print('yvec is')
            #print(yvec)
            ymode, count=mode(yvec)
            #print(ymode)
            #print(count)
        
           # print('File {} is predicted {}'.format(fn, instruments[ymode[0]]))
            #print(ymode)
            y.append(ymode[0])
            ytrue.append(_cls)
            files.append(fpath)
            #print(y)
            fnum+=1
            
    return y, files, ytrue #, MasterX, Masterytrue, Mastery




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default='validation',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='clean100full',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
   
    #downsample audio to 16 kHz becuase we dont care about higher frequencies
    #prior to downsampling, sample rate is around 44.1 kHz
    parser.add_argument('--sr', type=int, default=16000,
                        help='rate to downsample audio')

    #'3a3d0279' is third file in Hi_hatx
    parser.add_argument('--fn', type=str, default= '8da75280', #'3a3d0279',
                        help='file to plot over time to check magnitude')
    parser.add_argument('--threshold', type=str, default=100,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

 











## CONTINUE AQUINO CODE


#[y, files, ytrue, MasterX, Masterytrue, Mastery]=classify_files(args)
[y, files, ytrue]=classify_files(args)
ypredicted=[]
for count in range(len(y)):
    ypredicted.append(instruments[y[count]])
 #   Masterypredicted=[]
#for count in range(len(Mastery)):
#    Masterypredicted.append(instruments[Mastery[count]])
    

# predict labels
#Y_pred = estimator.predict(X)
#score = estimator.score(X, y)
total=0.0
for k in range(len(ypredicted)):
    if ypredicted[k]==ytrue[k]:
        total+=1.0
score = total/len(y)

#Test perfornmance
print("\n \n Model Accuracy: {:f} \n".format(score))

# Plot confusion matrix
#titles_options = [("Confusion matrix, without normalization", None),
#                  ("Normalized confusion matrix", 'true')]
#for title, normalize in titles_options:
#    disp = plot_confusion_matrix(estimator, MasterX, Masterytrue,
#                                 display_labels=instruments,
#                                 cmap=plt.cm.Blues,
#                                 normalize=normalize)
#    disp.ax_.set_title(title)

#plt.show()


for (file, label, truelabel) in zip(files,ypredicted, ytrue):
    print('File {:50} is predicted {:15} and is actually {:12}.'.format(file, label, truelabel))