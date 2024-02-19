import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import glob
import json
import math
import pdb

# pdb.set_trace()
glob = glob.glob("/home/arjay55/code/datasets/pcd_train/pointnet_ready_*.npy")
# get number of elements in glob list
n = len(glob)
# iterate to form an array of strings with the format "full{n}.npy"
for i in range(n):
    glob[i] = f"pointnet_ready_{i}.npy"


train_percent = 0.75

# import json file to pandas dataframe with no header
df = pd.read_csv('/home/arjay55/code/datasets/pcd_train/pose.json', sep=' ', header=None)   
#select last 4 columns with quaternion data of format qw qx qy qz and convert each row to matrix of shape (3,3)
dfquat = df.iloc[:math.floor(train_percent*n), -4:].apply(lambda x: R.from_quat(x).as_matrix(), axis=1)
# convert each numpy array to list
dfquat = dfquat.apply(lambda x: x.tolist())
dftranslation = df.iloc[:math.floor(train_percent*n), :3].values
dffocal = np.full((dftranslation.shape[0],2),3.0)
#create a json file with the following format where out_arr is a list with key "filepath"
# and dfquat, dftranslation, dffocal are numpy arrays with keys "R", "T", "focal_length" respectively

#enumerate glob
data = {}
apple = []
rot = dfquat.values.tolist()
tran = dftranslation.tolist()
foc = dffocal.tolist()
for count in range(math.floor(len(glob)*train_percent)):
    apple.append({"filepath":glob[count],"R":rot[count],"T":tran[count],"focal_length":foc[count],"principal_point":[-0.0, -0.0]})

data = {"apple":apple}
#save the json file
with open('/home/arjay55/code/datasets/pcd_train/pcd_train_train.json', 'w') as f:
    json.dump(data, f)


# import json file to pandas dataframe with no header
df = pd.read_csv('/home/arjay55/code/datasets/pcd_train/pose.json', sep=' ', header=None)   
#select last 4 columns with quaternion data of format qw qx qy qz and convert each row to matrix of shape (3,3)
dfquat = df.iloc[:, -4:].apply(lambda x: R.from_quat(x).as_matrix(), axis=1)
# convert each numpy array to list
dfquat = dfquat.apply(lambda x: x.tolist())
dftranslation = df.iloc[:, :3].values
dffocal = np.full((dftranslation.shape[0],2),3.0)
#create a json file with the following format where out_arr is a list with key "filepath"
# and dfquat, dftranslation, dffocal are numpy arrays with keys "R", "T", "focal_length" respectively

#enumerate glob
data = {}
apple = []
rot = dfquat.values.tolist()
tran = dftranslation.tolist()
foc = dffocal.tolist()
for count in range(math.floor(len(glob)*train_percent)+1,len(glob)):
    apple.append({"filepath":glob[count],"R":rot[count],"T":tran[count],"focal_length":foc[count],"principal_point":[-0.0, -0.0]})

data = {"apple":apple}
#save the json file
with open('/home/arjay55/code/datasets/pcd_train/pcd_train_test.json', 'w') as f:
    json.dump(data, f)
