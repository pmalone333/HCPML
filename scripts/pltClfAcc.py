import numpy as np
import os
import matplotlib.pyplot as plt

samp_sizes = [10,20,50,100,200,500,1080]
mvpa_path  = '/Volumes/maloneHD/Data/HCP_ML/motor/mvpa/lfvslh/'
cv_type    = 'LOSO'

accBySampSize = np.empty([len(samp_sizes),2])
semBySampSize = np.empty([len(samp_sizes),2])
for i, s in enumerate(samp_sizes):
    #load SVM results
    cv_results = np.load(os.path.join(mvpa_path,'cv_results',
                                      str(s)+'subs_'+cv_type+
                                      '_CV_SVMclfAcc.npy'))
    #average acc across CV folds
    acc_mean = np.mean(cv_results,1)
    #average/std acc across parcels
    pmean    = (np.mean(acc_mean)).round(2)
    pstd     = (np.std(acc_mean)).round(2)

    accBySampSize[i,0] = pmean
    semBySampSize[i, 0] = pstd/np.sqrt(360)

    #load KNN results
    cv_results = np.load(os.path.join(mvpa_path,'cv_results',
                                      str(s)+'subs_'+cv_type+
                                      '_CV_KNNclfAcc.npy'))
    #average acc across CV folds
    acc_mean = np.mean(cv_results,1)
    #average/std acc across parcels
    pmean    = (np.mean(acc_mean)).round(2)
    pstd     = (np.std(acc_mean)).round(2)

    accBySampSize[i,1] = pmean
    semBySampSize[i, 1] = pstd/np.sqrt(360)


plt.figure(figsize=(12, 9))

# remove plot frame lines
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
#
# ensure that the axis ticks only show up on the bottom and left of the plot
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.ylim(0.5, 0.65)

# make xticks larger enough to read
plt.xticks(range(0, 1100, 100), fontsize=14)

plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Sample size (nsubs)", fontsize=16)

# matplotlib's fill_between() call to create error bars.
# SVM error bars
plt.fill_between(samp_sizes, accBySampSize[:,0] - semBySampSize[:,0],
                 accBySampSize[:, 0] + semBySampSize[:, 0], color="#3F5D7D")

# KNN error bars
plt.fill_between(samp_sizes, accBySampSize[:,1] - semBySampSize[:,1],
                 accBySampSize[:, 1] + semBySampSize[:, 1], color="#2cf7b2")

# plot the means as a white line in between the error bars.
plt.plot(samp_sizes, accBySampSize[:,0], color="white", lw=2, label='SVM')
plt.plot(samp_sizes, accBySampSize[:,1], color="white", lw=2, label='KNN')

plt.legend(loc=3)

#change legend color to color of error bar
ax = plt.gca()
leg = ax.get_legend()
hl_dict = {handle.get_label(): handle for handle in leg.legendHandles}
hl_dict['SVM'].set_color(color="#3F5D7D")
hl_dict['KNN'].set_color(color="#2cf7b2")

plt.savefig(os.path.join(mvpa_path,'images','accBySampSize.png'),dpi=400)
