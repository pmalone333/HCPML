import numpy as np
import os
import matplotlib.pyplot as plt

samp_sizes = [10,20,50,100,200,500,1080]
mvpa_path  = '/Volumes/maloneHD/Data/HCP_ML/motor/mvpa/lfvslh/'
cv_type    = 'LOSO'

nBelowChnce = np.empty([len(samp_sizes),2])
for i, s in enumerate(samp_sizes):
    #load SVM results
    cv_results = np.load(os.path.join(mvpa_path,'cv_results',
                                      str(s)+'subs_'+cv_type+
                                      '_CV_SVMclfAcc.npy'))
    #average acc across CV folds
    acc_mean = np.mean(cv_results,1)
    #average/std acc across parcels
    #pmean    = (np.mean(acc_mean)).round(2)
    #pstd     = (np.std(acc_mean)).round(2)

    nBelowChnce[i,0] = sum(acc_mean<0.5)

    #load KNN results
    cv_results = np.load(os.path.join(mvpa_path,'cv_results',
                                      str(s)+'subs_'+cv_type+
                                      '_CV_KNNclfAcc.npy'))
    #average acc across CV folds
    acc_mean = np.mean(cv_results,1)
    #average/std acc across parcels
    #pmean    = (np.mean(acc_mean)).round(2)
    #pstd     = (np.std(acc_mean)).round(2)

    nBelowChnce[i,1] = sum(acc_mean<0.5)


plt.figure(figsize=(12, 9))

# remove plot frame lines
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
#
# ensure that the axis ticks only show up on the bottom and left of the plot
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.ylim(0, 100)

# make xticks larger enough to read 
plt.xticks(range(0, 1100, 100), fontsize=14)

plt.ylabel("Num ROI with below-chance clf acc", fontsize=16)
plt.xlabel("Sample size (nsubs)", fontsize=16)

# plot the means as a white line in between the error bars.
plt.plot(samp_sizes, nBelowChnce[:,0], color="#3F5D7D", lw=2, label='SVM')
plt.plot(samp_sizes, nBelowChnce[:,1], color="#2cf7b2", lw=2, label='KNN')

plt.legend(loc=1)

plt.savefig(os.path.join(mvpa_path,'images','accBelowChance.png'),dpi=200)
