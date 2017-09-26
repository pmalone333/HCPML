from mvpa2.suite import *

def runCV(p,sub_fds,clf,cv,nparc):

    #enable output to console
    verbose.level  = 2
    clf_start_time = time.time()
    cv_out         = cv(sub_fds)
    verbose(2, "classification computation time: %.1f seconds" % (time.time() - clf_start_time))
    verbose(2, "parcel " + str(p) + " of " + str(nparc))
    return cv_out