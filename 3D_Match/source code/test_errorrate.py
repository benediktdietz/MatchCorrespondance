from numpy import array
import numpy as np

descLogFile = '3dmatch.log' # List of descriptor distances per comparison
gtLogFile = 'test_label.log'; # Ground truth binary correspondence labels

with open(descLogFile, 'r') as f:
    f = f.readlines()
descDistPred = []
for line in f:
    line = line.strip() # remove the newline character at the end
    descDistPred.append(line)
descDistPred = array(descDistPred).astype(float) 

descDistPred = descDistPred[1:]

with open(gtLogFile, 'r') as f:
    f = f.readlines()
descIsMatchGT = []
for line in f:
    line = line.strip() # remove the newline character at the end
    descIsMatchGT.append(line)
descIsMatchGT = array(descIsMatchGT).astype(int) 
    
num_rest = descIsMatchGT[0]

    # Load ground truth binary labels (1 - is match, 0 - is nonmatch)
descIsMatchGT = descIsMatchGT[1:]  

    # Loop through 1000 descriptor distance thresholds
thresh = np.arange(0, max(descDistPred)*1.05,max(descDistPred)*1.05/1000)
    
allThresh = array([thresh,]*np.size(descIsMatchGT,0)).T
allDescDistPred = array([descDistPred,]*len(thresh))
allDescIsMatchGT = array([descIsMatchGT,]*len(thresh))
allDescIsMatchPred = (allDescDistPred < allThresh).astype(int)
  
numTP = np.sum(allDescIsMatchPred&allDescIsMatchGT,1)
numFP = np.sum(allDescIsMatchPred&(1-allDescIsMatchGT),1)
numTN = np.sum((1-allDescIsMatchPred)&(1-allDescIsMatchGT),1)
numFN = np.sum((1-allDescIsMatchPred)&allDescIsMatchGT,1)

np.seterr(divide='ignore', invalid='ignore')
accuracy = (numTP+numTN)/(numTP+numTN+numFP+numFN)
precision = numTP/(numTP+numFP)
recall = numTP/(numTP+numFN)
TNrate = numTN/(numTN+numFP)
FPrate = numFP/(numFP+numTN)

errorAt95Recall = np.mean(FPrate[(recall>0.949)&(recall<0.951)])
print('False-positive rate (error) at 95% recall: {0:4f}% \n'.format(errorAt95Recall*100));
