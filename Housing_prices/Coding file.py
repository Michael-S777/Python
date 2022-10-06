#Example of coding
#Prediction of housing prices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Load train data (Write train.csv directory)
train = pd.read_csv("C:/Users/Mikha/OneDrive/Desktop/Python/Housing_prices/train.csv")

#Load test data (Write test.csv directory)
test = pd.read_csv("C:/Users/Mikha/OneDrive/Desktop/Python/Housing_prices/test.csv")


#Make train set and test set in the same data set
data = train.append(test,sort=False)


#Visualize the DataFrame data
print(data)


#Plot features with more than 1000 NULL values

features = []
nullValues = []
for i in data:
    if (data.isna().sum()[i])>1000 and i!='SalePrice':
        features.append(i)
        nullValues.append(data.isna().sum()[i])
y_pos = np.arange(len(features)) 
plt.bar(y_pos, nullValues, align='center', alpha=0.5)
plt.xticks(y_pos, features)
plt.ylabel('NULL Values')
plt.xlabel('Features')
plt.title('Features with more than 1000 NULL values')
plt.show()

#Dropping columns that contain more than 1000 NULL values
data = data.dropna(axis=1, how='any', thresh = 1000)


#Replacing NULL values with mean values
data = data.fillna(data.mean())
print(data)


#Converting string values to integer values
data = pd.get_dummies(data)

#Dropping features that are correlated to each other
#If the correlation between the features is > 0.8
#Adding one of them to the set

covarianceMatrix = data.corr()
listOfFeatures = [i for i in covarianceMatrix]
setOfDroppedFeatures = set() 
for i in range(len(listOfFeatures)) :
    for j in range(i+1,len(listOfFeatures)):
        feature1=listOfFeatures[i]
        feature2=listOfFeatures[j]
        if abs(covarianceMatrix[feature1][feature2]) > 0.8:
            setOfDroppedFeatures.add(feature1)
 
data = data.drop(setOfDroppedFeatures, axis=1)


#Dropping features that are not correlated with output
nonCorrelatedWithOutput = [column for column in data if abs(data[column].corr(data["SalePrice"])) < 0.045]

data = data.drop(nonCorrelatedWithOutput, axis=1)


#Dealing with outliers
#Plotting one of the features with outliers
plt.plot(data['LotArea'], data['SalePrice'], 'bo')
plt.axvline(x=75000, color='r')
plt.ylabel('SalePrice')
plt.xlabel('LotArea')
plt.title('SalePrice in function of LotArea')
plt.show()


#First, we need to seperate the data (Because removing outliers ⇔ removing rows, and we don't want to remove rows from test set)
newTrain = data.iloc[:1460]
newTest = data.iloc[1460:]


#Second, we will define a function that returns outlier values using percentile() method

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75]) #Get 1st and 3rd quartiles (25% -> 75% of data will be kept)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5) #Get lower bound
    upper_bound = quartile_3 + (iqr * 1.5) #Get upper bound
    return np.where((ys > upper_bound) | (ys < lower_bound)) #Get outlier values

#Third, we will drop the outlier values from the train set

trainWithoutOutliers = newTrain #We can't change train while running through it

for column in newTrain:
    outlierValuesList = np.ndarray.tolist(outliers_iqr(newTrain[column])[0]) #outliers_iqr() returns an array
    trainWithoutOutliers = newTrain.drop(outlierValuesList) #Drop outlier rows

trainWithoutOutliers = newTrain


#Training the data
#Removing SalePrice column
X = trainWithoutOutliers.drop("SalePrice", axis=1) 
#Getting SalePrice column {log1p(x) = log(x+1)}
Y = np.log1p(trainWithoutOutliers["SalePrice"]) 
reg = LinearRegression().fit(X, Y)


#Making prediction
#Removing SalePrice column
newTest = newTest.drop("SalePrice", axis=1) 
pred = np.expm1(reg.predict(newTest))

#Submitting prediction
sub = pd.DataFrame() #Create a new DataFrame for submission
sub['Id'] = test['Id']
sub['SalePrice'] = pred
sub.to_csv("submission.csv", index=False) #Convert DataFrame to .csv file

#Visualizing the DataFrame sub
sub 
