import pgm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --------- Import faces ----------
path = 'archive/archive'
num_subjects = 40
num_faces_per_subject = 10
train_test_ratio = 0.6
variance_threshold = 0.8

train_set_size = int(num_subjects*num_faces_per_subject*train_test_ratio)
test_set_size = int(num_subjects*num_faces_per_subject*(1-train_test_ratio))
#compute the size of every image by checking the first
f=path+'/s1/1.pgm'
img = pgm.read_pgm(f)
size = img.shape[0]*img.shape[1] 



faces_all = np.zeros([num_subjects*num_faces_per_subject, size])
faces_train = np.zeros([train_set_size, size])
faces_test = np.zeros([test_set_size, size])
print('Importing data...')
for i in range(num_subjects): #for each subject
    subject_faces = np.zeros([num_faces_per_subject, size]) #array containing all faces of the subject
    for j in range(num_faces_per_subject): #for each face
        f=path+'/s'+str(i+1)+'/'+str(j+1)+'.pgm' #compose filepath
        img = pgm.read_pgm(f) #read the file (2D array of shape m by n)
        img = img.reshape(size) #reshape the img into a 1D array
        subject_faces[j,:] = img 
    
    faces_all[i*num_faces_per_subject:(i+1)*num_faces_per_subject,:] = subject_faces
    faces_train[i*int(num_faces_per_subject*train_test_ratio):(i+1)*int(num_faces_per_subject*train_test_ratio),:] = subject_faces[0:int(num_faces_per_subject*train_test_ratio),:]
    faces_test[i*int(num_faces_per_subject*(1-train_test_ratio)):(i+1)*int(num_faces_per_subject*(1-train_test_ratio)),:] = subject_faces[0:int(num_faces_per_subject*(1-train_test_ratio)),:]

# -------- Training phase ---------------
L =  faces_train.shape[0]
mean_face = faces_train.mean(axis = 0) #compute mean face
faces_train_center = faces_train-mean_face #centering the dataset
cov_mat_mod = 1/L * (faces_train_center @ faces_train_center.transpose()) #computing covariance matrix
print('Computing eigenvalues of covariance matrix...')
eig_val, eig_vec = np.linalg.eig(cov_mat_mod) #computing eigenvalues and eigenvectors
sort_mask = np.argsort(eig_val)[::-1] #sorting eigenvalues and eigenvectors in decreasing order
eig_val = eig_val[sort_mask]
eig_vec = eig_vec[:, sort_mask]

explained_variance_cum = (eig_val/eig_val.sum()).cumsum() #computing the cumulative explained variance
threshold_mask = explained_variance_cum<variance_threshold
num_pcs_kept = (threshold_mask).sum()
print('Using variance threshold: '+str(variance_threshold)+'. Number of PCs kept: '+str(num_pcs_kept)+'/'+str(train_set_size))
eigenfaces = faces_train_center.transpose() @ eig_vec[:,threshold_mask]
faces_train_projected = faces_train @ eigenfaces


print('DEBUG')