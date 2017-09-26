from mvpa2.suite import *
import os
import platform
import numpy as np
import nibabel as nib
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt

#enable output to console
verbose.level = 2

script_start_time = time.time()

#define paths
task      = 'WM' #motor, WM, gambling
clf_name  = 'faceVsPlace' #lfvslh, multiclass (all 5 movements)

data_path = os.path.join('/Volumes/maloneHD/Data/HCP_ML/', task)  # base directory (mac)
beta_path = os.path.join('/Volumes/maloneHD/Data_noSync/HCP_ML/', task)  # beta images

mvpa_path = os.path.join(data_path,'mvpa',clf_name)
parc_path = os.path.join(data_path,'parc') #parcellations

#analysis parameters
nsubs    = 10 #number of subjects
nparc    = 360 #number of parcels/ROIs
clf_type = 'SVM' #KNN, SVM
knn_k    = round(np.sqrt(nsubs)) #k-nearest-neighbor parameter
cv_type  = 'nfld' #split_half, LOSO (leave-one-subject-out), nfld (n-fold)
targets  = ['face','place']
pe_num   = ['16','17']
# targets  = ['lf','lh'] #targets to be classified
# pe_num   = ['2','3'] #parameter estimate numbers corresponding to targets

#define subjects and mask
subs       = os.listdir(beta_path)
subs_train = subs[:nsubs]
subs_test  = subs[nsubs:nsubs+10]
surf_mask  = np.ones([1,59412]) #mask for cortical surface nodes, not subcortical/cerebellum volumetric voxels
msk_path   = os.path.join(parc_path, 'Glasser_360.dtseries.nii')
msk        = nib.load(msk_path)
msk_data   = msk.get_data()
msk_data   = msk_data[0, 0, 0, 0, 0, 0:]  #last dimension contains parcel data

#load beta imgs
ds_all = []
for index, s in enumerate(subs_train):
    tds_beta_path = os.path.join(beta_path, s,
                                 'MNINonLinear', 'Results', 'tfMRI_'+task,
                                 'tfMRI_'+task+'_hp200_s2_level2.feat',
                                 'GrayordinatesStats')
    pe_paths = []
    for p in pe_num:
        pe_paths.append(os.path.join(tds_beta_path,
                                     'cope'+p+'.feat','pe1.dtseries.nii'))

    #ds = fmri_dataset(pe_paths,targets=targets,mask=surf_mask)
    ds = fmri_dataset(pe_paths, targets=targets)

    ds.sa['subject'] = np.repeat(index, len(ds))
    #ds.fa['parcel']  = msk_data
    ds_all.append(ds)
    verbose(2, "subject %i of %i loaded" % (index, nsubs))

fds = vstack(ds_all) #stack datasets

#classifier algorithm
if clf_type is 'SVM':
    clf = LinearCSVMC()
elif clf_type is 'SVM-rbf':
    clf = RbfCSVMC()
elif clf_type is 'KNN':
    clf = kNN(k=knn_k, voting='weighted')
#cross-validation algorithm
if cv_type is 'split_half':
    cv = CrossValidation(clf,
                         HalfPartitioner(count=2,
                                         selection_strategy='random', attr='subject'),
                         errorfx=mean_match_accuracy)
elif cv_type is 'LOSO':
    cv = CrossValidation(clf,
                         NFoldPartitioner(attr='subject'),
                         errorfx=mean_match_accuracy)
elif cv_type is 'nfld':
    cv = CrossValidation(clf,
                         NFoldPartitioner(count=5,
                                         selection_strategy='random', attr='subject'),
                         errorfx=mean_match_accuracy)
#run classification
parc       = range(1,nparc+1)
cv_results = [0 for x in parc]
num_cores  = multiprocessing.cpu_count()
#whole brain clf
cv_out = cv(fds)
#roi-wise clf
# cv_results = Parallel(n_jobs=num_cores)(delayed(runCV.runCV)
#                                         (p,fds[:, fds.fa.parcel == p],clf,cv,nparc) for p in parc)

#get feature weights
sensana = clf.get_sensitivity_analyzer()
sens    = sensana(fds)

#convert feature weights to numpy array and save
sens_out = np.asarray(sens)
np.save(os.path.join(mvpa_path,'cv_results',str(nsubs)+'subs_'+cv_type+'_CV_'+clf_type+'ftrWghts_faceVsPlace'),
        sens_out)

#feature weights x 2bk>0bk beta map
dp = np.zeros([len(subs_test),1])
for index, s in enumerate(subs_test):
    path = os.path.join(beta_path, s,
                                 'MNINonLinear', 'Results', 'tfMRI_'+task,
                                 'tfMRI_'+task+'_hp200_s2_level2.feat',
                                 'GrayordinatesStats','cope2.feat','pe1.dtseries.nii')
    beta_map  = nib.load(path)
    beta_map  = np.array(beta_map.dataobj)
    beta_map  = beta_map[0, 0, 0, 0, :, 0:]
    dp[index] = np.dot(sens_out,beta_map.transpose())

np.save(os.path.join(mvpa_path,'cv_results',str(nsubs)+'subs_'+cv_type+'_CV_'+clf_type+'dp_faceVsPlace'),
        dp)

#load behavioral data
df    = pd.read_csv('HCP_behavioraldata.csv')
subs  = [int(s) for s in subs_test] #convert str to int
df2   = df.loc[df['Subject'].isin(subs_test)]
bdata = df2.ListSort_AgeAdj
#bdata = df2.Dexterity_AgeAdj
bdata = bdata.reshape(300,1)

verbose(2, "total script computation time: %.1f minutes" % ((time.time() - script_start_time)/60))
