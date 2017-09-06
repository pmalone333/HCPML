import platform
import os
from pathlib import Path

#define paths
task      = 'WM' #motor, WM, gambling
if platform.node() == 'D-128-208-56-246.dhcp4.washington.edu':
#if platform.node() == 'Patricks-MacBook-Pro.local':
    data_path = os.path.join('/Volumes/maloneHD/Data/HCP_ML/', task)  # base directory (mac)
    beta_path = os.path.join('/Volumes/maloneHD/Data_noSync/HCP_ML/', task, 'betas/')  # beta images
else:
    data_path = os.path.join('/media/malone/maloneHD/Data/HCP_ML/', task)  # base directory (linux)
    beta_path = os.path.join('/media/malone/maloneHD/Data_noSync/HCP_ML/', task, 'betas/') #beta images

nsubs    = 1075
subs     = os.listdir(beta_path)
count = 0;

for index, s in enumerate(subs):

    file_path = Path(os.path.join(beta_path, s,
                                 'MNINonLinear', 'Results', 'tfMRI_'+task,
                                 'tfMRI_'+task+'_hp200_s2_level2.feat',
                                 'GrayordinatesStats','cope9.feat','pe1.dtseries.nii'))
    if file_path.exists():
        print(s+' exists')
    else:
        print(s+' does not exist')
        bash_command = 'rm -r '+ os.path.join(beta_path,s)
        os.system(bash_command)
        count += 1

print(count)