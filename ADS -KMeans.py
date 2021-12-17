import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
import psutil
import os



print("K-Means Classifier")
# Reads the train and test data

#train = pd.read_csv('1Million.csv', header=None)
#train = pd.read_csv('500K.csv', header=None)
train = pd.read_csv('./KDDTrain+.csv', header=None)
test = pd.read_csv('./KDDTest+.csv', header=None)
print("Train and Test Data read...")



# Reads "Field Names.csv". Use this to set the names of train and test data columns
columns = pd.read_csv('Field Names.csv', header=None)
columns.columns = ['name', 'type']
train.columns = columns['name']
test.columns = columns['name']



# Read Attack Types.csv
# Use this to create a mapping from attack types to final labels (Normal, Dos, R2L, Prob, U2R)
attackType = pd.read_csv('Attack Types.csv', header=None)
attackType.columns = ['Name', 'Type']
attackMap = {}



#find count of type from attackType
attackTypeCount = len(attackType['Type'].drop_duplicates())
attackNames = attackType['Type'].drop_duplicates().values.tolist()
print('There are ' +  str(attackTypeCount) + ' attack type')
print('Attacks name are:')
print(attackNames)



# Creates attackMap map which contains a mapping between attack type and the final label
for i in range(len(attackType)):
    attackMap[attackType['Name'][i]] = attackType['Type'][i]
print("Attack type mapping created...")



# Add a new variable called 'label' which contains the final label
train['label'] = train['attack_type'].map(attackMap)
test['label'] = test['attack_type'].map(attackMap)




# Transform the existing nominal variables into the integer coded variables using the LabelEncoder
for col in ['protocol_type', 'flag', 'service', 'label']:
    le = LabelEncoder()
    le1 = LabelEncoder()    
    le.fit(train[col])
    le1.fit(test[col])
    train[col] = le.transform(train[col])
    test[col] = le1.transform(test[col])
 
    
# The variable 'label' is stored in different variables
# This is required to keep the dependent variable separate from the independent variable
trainLabel = train['label']
testLabel = test['label']



# attack_type and label variables are removed from the train and test data
# so only features are remained in train and test data
train.drop(['attack_type', 'label'], axis=1, inplace=True)
test.drop(['attack_type', 'label'], axis=1, inplace=True)
print("Train and Test data labels created...")


# for Stadart Scaler   
scaler = MinMaxScaler()   #scale between 0 and 1
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)
total = np.concatenate([train, test] )



# Decomposition features are generated for both train and test data
pca = PCA(n_components=41, random_state=100)
pca.fit(total)
train = pca.transform(train)
test = pca.transform(test)


print("Decomposed features created...")
print("Number of features used : %d" % train.shape[1])



# Performing k-fold Cross validation
startTime = time.clock()
KM1 = KMeans(n_clusters = 5, init = 'random', random_state=100)
KM1.fit(train, trainLabel)
score = abs(cross_val_score(KM1, train, trainLabel, cv=5))/1000
endTime = time.clock()
print("Time taken to perform 5-fold cross validation : %f" % (endTime - startTime))
print("Cross validation scores : ")
print(score)
print("Mean score of 5-fold cross validation : %f" % abs(score.mean()/1000))



# Final Testing and Evaluate Performance
# Train the KNN classifier model by original train data and got optimized parameter
startTime = time.clock()
KM2 = KMeans(n_clusters = 5, init = 'k-means++' ) 
KM2.fit(train, trainLabel)
endTime = time.clock()
print("Time taken to train final model : %f" % (endTime - startTime))
print("Predictions made using final model...")



# Predictions for test data and evaluate its performance
startTime = time.clock()
pred = KM2.predict(test)
endTime = time.clock()
cpuUsage = psutil.cpu_percent()
pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info()[0] / 2. ** 30
print("Time taken to make predictions on test data : %f" % (endTime - startTime))
print("Memory used : %f GB  CPU usage : %f" % (memoryUse, cpuUsage))


#calculate scores and confusion_matrix
Classes = ['normal', 'probe', 'dos', 'r2l','u2r' ]
acc = accuracy_score(y_pred=pred, y_true=testLabel)
con_matrix = confusion_matrix(y_pred=pred, y_true=testLabel, labels= list(le.transform(Classes))) #, labels=Classes
print("Confusion matrix : ")
print(con_matrix)


# Print accuracy and detection rate
acc = accuracy_score(y_pred=pred, y_true=testLabel)
print("Accuracy score on test data is : %f" % acc)

sumDr = 0
for i in range(con_matrix.shape[0]):
    det_rate = 0
    for j in range(con_matrix.shape[1]):
        if i != j :
            det_rate += con_matrix[i][j]
    if con_matrix[i][i] != 0 or (det_rate + con_matrix[i][i])  != 0:
        det_rate =100* con_matrix[i][i]/(det_rate + con_matrix[i][i])
        sumDr += det_rate
        print("For " + Classes[i] + ", Detection Rate is % " + str(det_rate))
    else:
        print("For " + Classes[i] + ", Detection Rate is % 0")

DR = sumDr/attackTypeCount
print("DR is % " + str(DR))