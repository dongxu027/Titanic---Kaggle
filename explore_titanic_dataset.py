"""
The first part is an exercise to explore the training data set with Pandas
"""

import pandas as pd
import pdb

train_data = pd.read_csv('train.csv')

train_data.head(n=10)

#    PassengerId  Survived  Pclass  \
# 0            1         0       3   
# 1            2         1       1   
# 2            3         1       3   
# 3            4         1       1   
# 4            5         0       3   
# 5            6         0       3   
# 6            7         0       1   
# 7            8         0       3   
# 8            9         1       3   
# 9           10         1       2   

#                                                 Name     Sex  Age  SibSp  \
# 0                            Braund, Mr. Owen Harris    male   22      1   
# 1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female   38      1   
# 2                             Heikkinen, Miss. Laina  female   26      0   
# 3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female   35      1   
# 4                           Allen, Mr. William Henry    male   35      0   
# 5                                   Moran, Mr. James    male  NaN      0   
# 6                            McCarthy, Mr. Timothy J    male   54      0   
# 7                     Palsson, Master. Gosta Leonard    male    2      3   
# 8  Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)  female   27      0   
# 9                Nasser, Mrs. Nicholas (Adele Achem)  female   14      1   

#    Parch            Ticket     Fare Cabin Embarked  
# 0      0         A/5 21171   7.2500   NaN        S  
# 1      0          PC 17599  71.2833   C85        C  
# 2      0  STON/O2. 3101282   7.9250   NaN        S  
# 3      0            113803  53.1000  C123        S  
# 4      0            373450   8.0500   NaN        S  
# 5      0            330877   8.4583   NaN        Q  
# 6      0             17463  51.8625   E46        S  
# 7      1            349909  21.0750   NaN        S  
# 8      2            347742  11.1333   NaN        S  
# 9      0            237736  30.0708   NaN        C  

# next provide brief data summary for each column, 
# the first thing to check is if there is any missing value:


# >>> train_data.isnull().sum()
# PassengerId      0
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age            177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         2

# we find that age and cabin has a lot of missing values. We assume these info can be critical in prediction

# step 1 is to do missing value imputation, look at Age first.

# note that in name there is may be some prefix, and those will be are related to Age
nameVect = list(train_data['Name'])

nameTitle = []
for name in nameVect:
  if "Mr. " in name:
    nameTitle.append("mr")
  elif "Mrs. " in name:
    nameTitle.append("mrs")
  elif "Miss. " in name:
    nameTitle.append("mis")
  else:
    nameTitle.append("na")

nameTitleSeries = pd.Series(nameTitle)
# use the value_counts() method to get the frequency count
# >>> nameTitleSeries.value_counts()
# mr     517
# mis    182
# mrs    125
# na      67
# from above, we created a simple imputed_age column

ageWhereNameContainsMr = train_data[nameTitleSeries == 'mr']['Age']
ageWhereNameContainsMrs = train_data[nameTitleSeries == 'mrs']['Age']
ageWhereNameContainsMis = train_data[nameTitleSeries == 'mis']['Age']
ageWhereNameContainsNa = train_data[nameTitleSeries == 'na']['Age']

# use decribe method to print out basic data summary
ageWhereNameContainsMr.describe()
# >>> ageWhereNameContainsMr.describe()
# count    398.000000
# mean      32.368090
# std       12.708793
# min       11.000000
# 25%       23.000000
# 50%       30.000000
# 75%       39.000000
# max       80.000000
# Note that describe method disregard 'NaN' values
inferredAgeWhereNameContainsMr = ageWhereNameContainsMr.median()
inferredAgeWhereNameContainsMrs = ageWhereNameContainsMrs.median()
inferredAgeWhereNameContainsMis = ageWhereNameContainsMis.median()
inferredAgeWhereNameContainsNa = ageWhereNameContainsNa.median()

import numpy as np

imputedAge = np.copy(train_data['Age'])

for iter_index in range(len(imputedAge)):
  if np.isnan(imputedAge[iter_index]):
    if nameTitleSeries[iter_index] == 'mr':
      imputedAge[iter_index] = inferredAgeWhereNameContainsMr
    elif nameTitleSeries[iter_index] == 'mrs':
      imputedAge[iter_index] = inferredAgeWhereNameContainsMrs
    elif nameTitleSeries[iter_index] == 'mis': 
      imputedAge[iter_index] = inferredAgeWhereNameContainsMis
    else:
      imputedAge[iter_index] = inferredAgeWhereNameContainsNa


# the other variable with lots of missing data is cabin, we make two assumptions
# people with same last name, stay in the same cabin
# cabin number could be related to price

# 1. find out price distribution of each cabin for non-missing data

cabinVect = list(train_data['Cabin'])
cabinMedianPrice = {}
medianPriceList = []

uniqueCabin = list(set(cabinVect))
print "checking unique cabin data"
print uniqueCabin

isCabinInfoNan = pd.isnull(train_data['Cabin'])

for cabin_info in uniqueCabin:
  # use isnull to check for NaN value
  if pd.isnull(cabin_info) == False:
    subset_df = train_data[train_data['Cabin']==cabin_info]
    medianPrice = subset_df['Fare'].median()
    cabinMedianPrice[cabin_info] = medianPrice
    medianPriceList.append(medianPrice)

print "checking medianPriceList"
print cabinMedianPrice

cabinWithImputation = []

# insert breakpoint
for index, cabin_info in enumerate(cabinVect):
  if pd.isnull(cabin_info) == True:
    fare_of_missing_cabin = train_data.iloc[index,9]
    # print index, cabin_info, fare_of_missing_cabin
    abs_diff_vect = list(abs(np.array(medianPriceList) - fare_of_missing_cabin))
    nearest_value = uniqueCabin[abs_diff_vect.index(min(abs_diff_vect))]
    cabinWithImputation.append(nearest_value)
  else:
    cabinWithImputation.append(cabin_info)

# print "print cabin info with imputation"
# print cabinWithImputation

train_data['cabinWithImputation'] = cabinWithImputation
train_data['imputedAge'] = imputedAge

print train_data.head(n=10)
print train_data.isnull().sum()

# # to get people within the same family, first extract the last name
# lastName = [item.split(',')[0] for item in nameVect]
# # add new column of lastname to the dataframe
# train_data['lastname'] = lastName

# for name in set(lastName):
#   subset_df = train_data[train_data['lastname']==name]
#   if subset_df.shape[0]>1:
#     print subset_df
#     break

filtered_train_data=train_data[pd.isnull(train_data['Embarked'])!=True]

df_dummied_gender = pd.get_dummies(filtered_train_data['Sex'])
df_dummied_cabin = pd.get_dummies(filtered_train_data['cabinWithImputation'])
df_dummied_embark = pd.get_dummies(filtered_train_data['Embarked'])

design_matrix = pd.concat([pd.DataFrame(filtered_train_data['Pclass']), 
  pd.DataFrame(filtered_train_data['imputedAge']), 
  pd.DataFrame(filtered_train_data['SibSp']),
  pd.DataFrame(filtered_train_data['Fare']),
  df_dummied_gender, df_dummied_cabin, df_dummied_embark], axis=1)

print 'Training'
from sklearn.ensemble import RandomForestClassifier 
forest = RandomForestClassifier(n_estimators=100)

from sklearn import cross_validation
for rep_index in range(50):
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(design_matrix, filtered_train_data['Survived'], test_size=0.05)
  forest = forest.fit(X_train, y_train)
  # print "checking model performance, by looking at the cross-validation"
  print forest.score(X_test,y_test)