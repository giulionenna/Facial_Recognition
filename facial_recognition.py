import pgm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize

# --------- Import faces ----------
path = 'archive/archive'
num_subjects = 39 #leave one out for out-of-training experiment
left_out = True #take the last subject for threshold assessment
num_faces_per_subject = 10
train_test_ratio = 0.6
variance_threshold_vec = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
acceptance_threshold =  6*10**3
acceptance_tolerance = 1.3
plot_flag = False

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
    faces_train[i*int(num_faces_per_subject*train_test_ratio):(i+1)*int(num_faces_per_subject*train_test_ratio),:] = \
        subject_faces[0:int(num_faces_per_subject*train_test_ratio),:]
    faces_test[i*int(num_faces_per_subject*(1-train_test_ratio)):(i+1)*int(num_faces_per_subject*(1-train_test_ratio)),:] = \
          subject_faces[int(num_faces_per_subject*train_test_ratio):num_faces_per_subject,:]

#--- importing the subject left out of the dataset---
if left_out:
    left_out_subject = np.zeros([num_faces_per_subject, size])
    for j in range(num_faces_per_subject):
        f=path+'/s'+str(num_subjects+1)+'/'+str(j+1)+'.pgm'
        img = pgm.read_pgm(f)
        img= img.reshape(size)
        left_out_subject[j,:] = img

# -------- Training phase ---------------
L =  faces_train.shape[0]
mean_face = faces_train.mean(axis = 0) #compute mean face
faces_train_center = faces_train-mean_face #centering the dataset (Phi^T)
cov_mat_mod = 1/L * (faces_train_center @ faces_train_center.transpose()) #computing modified covariance matrix
print('Computing eigenvalues of covariance matrix...')
eig_val, eig_vec = np.linalg.eig(cov_mat_mod) #computing eigenvalues and eigenvectors
sort_mask = np.argsort(eig_val)[::-1] #sorting eigenvalues and eigenvectors in decreasing order
eig_val = eig_val[sort_mask]
eig_vec = eig_vec[:, sort_mask]

explained_variance_cum = (eig_val/eig_val.sum()).cumsum() #computing the cumulative explained variance

count = 0
accuracy = np.zeros(len(variance_threshold_vec))
for variance_threshold in variance_threshold_vec:
    threshold_mask = explained_variance_cum<variance_threshold
    num_pcs_kept = (threshold_mask).sum()
    print('Using variance threshold: '+str(variance_threshold)+'\t Number of PCs kept: '+str(num_pcs_kept)+'/'+str(train_set_size))
    eigenfaces = faces_train_center.transpose() @ eig_vec[:,threshold_mask] #computing eigenvalues of the original covariance matrix
    eigenfaces = normalize(eigenfaces, axis=0, norm='l2') #normalize by columns the eigenvectors (eigenfaces is now an orthonormal matrix)
    faces_train_projected = faces_train_center @ eigenfaces
    plt.imshow(eigenfaces[:,0].reshape([112,92]), cmap='gray')

    #--------- TEST PHASE -------------
    faces_test_centered = faces_test-mean_face
    faces_test_projected = faces_test_centered @ eigenfaces #project test faces onto eigenspace
    faces_test_projected_back = faces_test_projected @ eigenfaces.transpose() #project back onto face space
    distance_from_face_space = np.linalg.norm(faces_test_centered-faces_test_projected_back, axis=1) #compute distance from eigenspace for each face

    #---acceptance threshold assessment---
    left_out_subject_centered = left_out_subject - mean_face
    left_out_subject_projected = left_out_subject_centered @ eigenfaces #project left out subject face onto eigenspace
    left_out_subject_projected_back = left_out_subject_projected @ eigenfaces.transpose() #project back onto face space
    left_out_distance = np.linalg.norm(left_out_subject_centered - left_out_subject_projected_back, axis=1)
    acceptance_threshold = acceptance_tolerance*left_out_distance.mean() #distance of a face not present among training subjects

    #---prediction---
    predicted = -1*np.ones(test_set_size) #init prediction vector
    true_faces = np.zeros(test_set_size) #init true classification idx
    test_faces_per_subject = int(num_faces_per_subject*(1-train_test_ratio))
    for i in range(num_subjects):
        true_faces[i*test_faces_per_subject:(i+1)*test_faces_per_subject] = i

    if(plot_flag and num_pcs_kept>1):
        train_idx = np.zeros(train_set_size)
        train_faces_per_subject = int(num_faces_per_subject*(train_test_ratio))
        for i in range(num_subjects):
            train_idx[i*train_faces_per_subject:(i+1)*train_faces_per_subject] = i

        plt.scatter(faces_train_projected[:,0], faces_train_projected[:,1], c=train_idx)
        plt.scatter(faces_test_projected[:,0], faces_test_projected[:,1], c=true_faces, marker='x')
        plt.show()

    #Prediction
    for i in range(test_set_size):#for each face in the test set
        face = faces_test_projected[i]
        if distance_from_face_space[i]>acceptance_threshold: #if the face is too far from eigenspace
            continue #skip face
        dist = np.linalg.norm(faces_train_projected - face, axis=1) #compute distance of the face with every other face in the eigenspace
        idx = np.argmin(dist) #find the nearest face
        predicted[i] = idx // int(num_faces_per_subject*train_test_ratio) #predict using idx of the nearest face

    accuracy[count] = sum(true_faces == predicted)/test_set_size
    unrecognized = sum(predicted == -1)
    print('Accuracy obtained: '+str(accuracy[count])+'\t Recognized faces: '+str(test_set_size-unrecognized)+'/'+str(test_set_size))
    count += 1

plt.plot(variance_threshold_vec, accuracy)
plt.show()
        
    