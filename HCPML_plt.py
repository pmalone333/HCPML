import numpy as np
import os
import matplotlib.pyplot as plt

def clfAccHist(nsubs,clf_type,cv_type,chance,mvpa_path):

    #load clf results
    cv_results = np.load(os.path.join(mvpa_path,'cv_results',
                                      str(nsubs)+'subs_'+cv_type+
                                      '_CV_'+clf_type+'clfAcc.npy'))
    #average acc across CV folds
    acc_mean = np.mean(cv_results,1)
    #average/std acc across parcels
    pmean    = (np.mean(acc_mean)).round(2)
    pstd     = (np.std(acc_mean)).round(2)

    #plot acc histogram
    plt.figure(figsize=(8,6))
    plt.hist(acc_mean)
    plt.ylabel('Num parcels')
    plt.xlabel('Accuracy')
    plt.axis([0, 1, 0, 140])
    plt.axvline(chance, color='k', linestyle='dashed', linewidth=1)
    plt.title(str(nsubs)+' subs,'+cv_type+' CV, '+clf_type+
              ' clf: mean='+str(pmean)+' std='+str(pstd))

    plt.axvline(pmean, color='r', linestyle='dashed', linewidth=1)
    plt.savefig(os.path.join(mvpa_path,'images',str(nsubs)+'subs_'+cv_type+'CV_'+clf_type+'clfAcc.png'),dpi=400)
    return
