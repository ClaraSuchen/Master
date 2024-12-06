# -*- coding: utf-8 -*-
"""MasterCode20241204.ipynb

## Load the data
"""

import pandas as pd
from pathlib import Path



data_dir='C:...Dataset/WISDM'

#name columns for two locations
accel_columns = ['user', 'activity', 'timestamp', 'accel_x', 'accel_y', 'accel_z']
gyro_columns = ['user', 'activity', 'timestamp', 'gyro_x', 'gyro_y', 'gyro_z']

                                                                                        #function to looop over files in directory
def load_files_from_directories(dirs):
    all_files = []
    for directory in dirs:
        path = Path(directory)
        all_files.extend(path.glob('*'))                                                # Load all files in the directory
    return all_files

                                                                                        #call locations and files
directories = [data_dir+'/accel', data_dir+'/gyro']
files = load_files_from_directories(directories)

                                                                                        #list for the dataframes
accel_dfs = []
gyro_dfs = []

                                                                                        #fill dataframes
for idx, directory in enumerate(directories):
    for file in Path(directory).glob('*'):
        if idx == 0:                                                                    # For accelerometer files
            df = pd.read_table(file, sep=',', header=None, names=accel_columns)
            accel_dfs.append(df)
        elif idx == 1:                                                                  # For gyroscope files
            df = pd.read_table(file, sep=',', header=None, names=gyro_columns)
            gyro_dfs.append(df)

                                                                                        #combine files for two dataframes each
full_accel_data = pd.concat(accel_dfs, ignore_index=True)
full_gyro_data = pd.concat(gyro_dfs, ignore_index=True)

                                                                                        #how long are the file?
print("shape file full accel data",full_accel_data.shape)
print("shape file full gyro data ",full_gyro_data.shape)

                                                                                        #merge to one on user, activity and timestamp
merged_data = pd.merge(full_accel_data, full_gyro_data, on=['user', 'activity', 'timestamp'], how='inner')

                                                                                        #check how many of the files could be matched:
matched_rows_count = merged_data.shape[0]
print(f"Number of matched rows: {matched_rows_count}")


                                                                                        #name final columns
final_columns = ['user',  'activity','timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
merged_data = merged_data[final_columns]

print(merged_data)

                                                                                        #check matched files
total_accel_rows = full_accel_data.shape[0]
total_gyro_rows = full_gyro_data.shape[0]

                                                                                        # Number of matched rows
matched_rows_count = merged_data.shape[0]

                                                                                        # Calculate the percentage of matched rows based on full accel data
matched_percentage_accel = (matched_rows_count / total_accel_rows) * 100

                                                                                        #  Calculate the percentage of matched rows based on full gyro data
matched_percentage_gyro = (matched_rows_count / total_gyro_rows) * 100
                                                                                        # print
print(f"Total rows in accelerometer data: {total_accel_rows}")
print(f"Total rows in gyroscope data: {total_gyro_rows}")
print(f"Matched rows count: {matched_rows_count}")
print(f"Percentage of matched rows for accelerometer data: {matched_percentage_accel:.2f}%")
print(f"Percentage of matched rows for gyroscope data: {matched_percentage_gyro:.2f}%")

"""#checking the files"""

                                                                                        #counting the files for participants

count=0
for idx, directory in enumerate(directories):
    for file in Path(directory).glob('*'):

      count=count+1                                                                     #checking the amount of files: 51gyro+51accel
print(count)
                                                                                        #short check of different users
print(merged_data['user'].unique())                                                     #merged 51

"""## selected activites from the dataset"""

                                                                                        #select activities (walking, running, stairs, sitting, standing, kicking)
merged_data.info()
move6=merged_data[merged_data['activity'].isin(['A','B','C','D','E','M'])]
move6.info()                                                                           #print type

"""# z_axis from object to float"""

                                                                                        # the last column was separated with ; and will be removed
move6['accel_z'] = move6['accel_z'].map(lambda x: x.rstrip(';'))
move6['accel_z'] = move6['accel_z'].astype(float)                                       # and datatype transformed to float like other time series
move6['gyro_z'] = move6['gyro_z'].map(lambda x: x.rstrip(';'))
move6['gyro_z'] = move6['gyro_z'].astype(float)
move6.info()

"""## training test split"""

                                                                                        #packages
from sklearn.model_selection import train_test_split
import numpy as np

                                                                                        #three dataframes
train_data = pd.DataFrame()
val_data = pd.DataFrame()
test_data = pd.DataFrame()

                                                                                       # iterate through users
for user in move6['user'].unique():
    user_data = move6[merged_data['user'] == user]

                                                                                       # iterate through activities for each user
    for activity in user_data['activity'].unique():
        activity_data = user_data[user_data['activity'] == activity]

                                                                                        #  sorted by timestamp (important to keep sorted order for ts)
        activity_data = activity_data.sort_values(by='timestamp')

                                                                                        # calculate split index per person per activity
        train_index= int(len(activity_data)*0.6)
        val_index= int(len(activity_data)*0.8)

                                                                                        # split data by index
        train_split= activity_data.iloc[:train_index]
        val_split = activity_data.iloc[train_index:val_index]
        test_split = activity_data.iloc[val_index:]

                                                                                       # append to train, validation and test dataframes
        train_data = pd.concat([train_data, train_split], ignore_index=True)
        val_data = pd.concat([val_data,val_split],ignore_index=True)
        test_data = pd.concat([test_data, test_split], ignore_index=True)
                                                                                       # print overview
print("Training Data:")
print(train_data.head())
print("Validation Data:")
print(val_data.head())
print("Testing Data:")
print(test_data.head())

"""## visualize"""

import matplotlib.pyplot as plt                                                         # packages

"""# visualization of training test split"""

                                                                                        # data sample for one activity and one participant for three parts of the same ts
filteredtrain=train_data[(train_data['activity']=='A')&(train_data['user']==1601)]
filteredval=val_data[(val_data['activity']=='A')&(val_data['user']==1601)]
filteredtest=test_data[(test_data['activity']=='A')&(test_data['user']==1601)]
                                                                                        # plot accelerometer data on timestamp
plt.plot(filteredtrain['timestamp'],filteredtrain['accel_x'],label='train')
plt.plot(filteredval['timestamp'], filteredval['accel_x'], label='validation')
plt.plot(filteredtest['timestamp'],filteredtest['accel_x'],label='test')
plt.legend()
plt.title("The example of participant 2 (1601) - Walking")
plt.xlabel("timestamp")
plt.ylabel("accel_x")
plt.show()

"""# visualize
A=Walking, B=jogging, C=Stairs, D=Sitting, E=Standing, M=kicking
"""

                                                                                        # visualize activities with example accelerometer x train data for one participant (nr2) 1601

fig,ax=plt.subplots(2,3)
ax=ax.flatten()
activities= train_data['activity'].unique()                     #
for i, activity in enumerate(activities):                                               # loop over activities to make a graph for each
                                                                                        #filter data for user
  filtered=train_data[(train_data['activity']==activity)&(train_data['user']==1601)]

  ax[i].plot(filtered.index,filtered['accel_x'] )

plt.tight_layout()
plt.show()

"""## check missings"""

missi=train_data.isna().sum()
print(missi)

                                                                                        # copy of dataframe for next step
train_norm = train_data.copy()
val_norm = val_data.copy()
test_norm = test_data.copy()

"""## Features"""

coordinates = ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z']         # name of columns with raw data

# mean of windows over accel x,y,z, gyro x,y,z for train,val and test data
for dat in coordinates:                                                         # loop over all raw time series
  train_norm[f'{dat}_mean_rol']=train_norm[dat].rolling(window=120).mean()      # rolling mean for training data
  val_norm[f'{dat}_mean_rol']=val_norm[dat].rolling(window=120).mean()          # rolling mean for validation data
  test_norm[f'{dat}_mean_rol']=test_norm[dat].rolling(window=120).mean()        # rolling mean for test data

# standard deviation of windows over accel x,y,z, gyro x,y,z for train,val and test data
for dat in coordinates:
  train_norm[f'{dat}_std_rol']=train_norm[dat].rolling(window=120).std()
  val_norm[f'{dat}_std_rol']=val_norm[dat].rolling(window=120).std()
  test_norm[f'{dat}_std_rol']=test_norm[dat].rolling(window=120).std()

#checking the amount of variables 3 + 6 + 6 + 6 = 21 (descriptives, x,y,z,x,y,z, mean x,y,z,x,y,z, std x,y,z,x,y,z)
print(train_norm.shape)             # yes

# variance
for dat in coordinates:
  train_norm[f'{dat}_var_rol']=train_norm[dat].rolling(window=120).var()
  val_norm[f'{dat}_var_rol']=val_norm[dat].rolling(window=120).var()
  test_norm[f'{dat}_var_rol']=test_norm[dat].rolling(window=120).var()

# skewness
for dat in coordinates:
  train_norm[f'{dat}_skew_rol']=train_norm[dat].rolling(window=120).skew()
  val_norm[f'{dat}_skew_rol']=val_norm[dat].rolling(window=120).skew()
  test_norm[f'{dat}_skew_rol']=test_norm[dat].rolling(window=120).skew()

# kurtosis

for dat in coordinates:
  train_norm[f'{dat}_kurt_rol']=train_norm[dat].rolling(window=120).kurt()
  val_norm[f'{dat}_kurt_rol']=val_norm[dat].rolling(window=120).kurt()
  test_norm[f'{dat}_kurt_rol']=test_norm[dat].rolling(window=120).kurt()

# Root Mean Square of windows

for dat in coordinates:
  train_norm[f'{dat}_rms_rol']=np.sqrt(train_norm[dat].pow(2).rolling(window=120).mean())
  val_norm[f'{dat}_rms_rol']=np.sqrt(val_norm[dat].pow(2).rolling(window=120).mean())
  test_norm[f'{dat}_rms_rol']=np.sqrt(test_norm[dat].pow(2).rolling(window=120).mean())

"""## FFT Entropy and DFA"""

# Spectral Entropy with .rolling()

def spectral_entropy(time_series):                                                # a function for:
                                                                                  # Apply FFT
    fft = np.fft.fft(time_series)
                                                                                  # Compute Power Spectrum
    power_spectrum = np.abs(fft)**2
                                                                                  # Normalize Power Spectrum
    power_spectrum_normalized = power_spectrum / np.sum(power_spectrum)
                                                                                  # Compute Spectral Entropy  (negative sum of normalized power spectrum multiplied by the logarithm of the normalized powerspectrum)
    spectral_entropy_value = -np.sum(power_spectrum_normalized[power_spectrum_normalized > 0] * np.log2(power_spectrum_normalized[power_spectrum_normalized > 0]))

    return spectral_entropy_value
                                                                                  # set size rolling window
window_size = 120

for dat in coordinates:                                                           # calculate function (Spectral Entropy) over the time series
  train_norm[f'{dat}_SpecEnt_rol'] = train_norm[dat].rolling(window=window_size).apply(lambda x: spectral_entropy(x.values), raw=False) # for trainings data
  val_norm[f'{dat}_SpecEnt_rol'] = val_norm[dat].rolling(window=window_size).apply(lambda x: spectral_entropy(x.values), raw=False)     # for validation data
  test_norm[f'{dat}_SpecEnt_rol'] = test_norm[dat].rolling(window=window_size).apply(lambda x: spectral_entropy(x.values), raw=False)

print("Rolling Spectral Entropy:")                                                # short overview over Spectral Entropy
print(train_norm['accel_x_SpecEnt_rol'].describe())

"""# Scaling features before DFA and after for the rest
DFA is sensitive to scales, the variables are different and for the rest it is more info when scaled later
"""

from sklearn.preprocessing import MinMaxScaler

train_without_objects = train_norm.copy()                                        # make a df for all non descriptives and label

                                                                                 # Drop the specified columns
without_objects = train_without_objects.drop(columns=['user', 'activity', 'timestamp'])
floatfeatures=without_objects.columns.tolist()
                                                                                 # Display the modified DataFrame
print(floatfeatures)

train_scale = train_norm.copy()                                                  # df for each each data sample separate
val_scale = val_norm.copy()
test_scale = test_norm.copy()

scaler = MinMaxScaler()                                                          # actually min max scale

train_scale[floatfeatures] = scaler.fit_transform(train_scale[floatfeatures])    # fitting on training data
print('train scale: ', train_scale)

                                                                                 # same transformation for val
val_scale[floatfeatures] = scaler.transform(val_scale[floatfeatures])            # print and check
print('val scale: ', val_scale)

                                                                                 # scale test data
test_scale[floatfeatures] = scaler.transform(test_scale[floatfeatures])

"""DFA"""

"""The idea:
"In each bin, a least squares regression
is fit and subtracted within each window. Residuals are squared and
averaged within each window. Then, the square root is taken of the
average squared residual across all windows of a given size. This
process repeats for larger window sizes, growing by, say a power of 2,
up to $N/4$, where $N$ is the length of the series. In a final step, the
logarithm of those scaled root mean squared residuals (i.e.,
fluctuations) is regressed on the logarithm of window sizes. "

https://github.com/travisjwiltshire/fractal_regression_manuscript/blob/main/fractal_regression_paper_brm.Rmd

The attempt of implentation:
"""

#DFA on all accel_x_y_z gyro_x_y_z, window size changes, same transformations for val and test

def detrended_fluctuation_analysis(time_series, min_window=16, max_window=1024):                        # start function
    N = len(time_series)                                                                                # length
    time_series_cumsum = np.cumsum(time_series - np.mean(time_series))                                  # Cumulatively sum of the detrended time series
    windows = np.arange(min_window, max_window + 1, step=1)                                             # Window sizes
    F = []                                                                                              # List of fluctuations

    for window in windows:
        bins = np.array([time_series_cumsum[i:i + window] for i in range(0, N-window, window)])         # array with bins
                                                                                                        # Detrend each bin and calculate the root mean square fluctuation compared to a polynomial
        rms = [np.sqrt(np.mean(np.square(bin - np.polyval(np.polyfit(np.arange(len(bin)), bin, 1), np.arange(len(bin)))))) for bin in bins]
        F.append(np.mean(rms))                                                                          # add fluctuation to the list
    return windows, F

                                                                                                        # apply function on raw (scaled) data
for dat in coordinates:
    train_scale[f'{dat}_DFA'] = train_scale[dat].values
    windows, F = detrended_fluctuation_analysis(train_scale[f'{dat}_DFA'])

    plt.loglog(windows, F, label=dat)

# Finalizing the plot
plt.xlabel('Window size (log scale)')
plt.ylabel('Fluctuation function F(n) (log scale)')
plt.title('Detrended Fluctuation Analysis')
plt.legend()
plt.show()


print(train_scale['accel_x_DFA'])


                                                                    # DFA for validation set

for dat in coordinates:
    val_scale[f'{dat}_DFA'] = val_scale[dat].values
    windows, F = detrended_fluctuation_analysis(val_scale[f'{dat}_DFA'])

                                                                    # DFA for test set

for dat in coordinates:
    test_scale[f'{dat}_DFA'] = test_scale[dat].values
    windows, F = detrended_fluctuation_analysis(test_scale[f'{dat}_DFA'])

"""# checks
missing, shape, describe
"""

                                                                    # check train, val, test shape
print(train_scale.shape)
print(val_scale.shape)
print(test_scale.shape)

missi2=train_scale.isnull().sum()
print(missi2)

"""# exclude missings due to windowing"""

train_scale.info()

                                                                    # drop first 120 because of window transforming some values are left
train_scale=train_scale.iloc[120:]
val_scale=val_scale.iloc[120:]
test_scale = test_scale.iloc[120:]


                                                                     # to check ratio to train data
print((120 / len(train_scale)) * 100 )
print((120 / len(test_data)) * 100   )                                # and test
                                                                      # removing produced missings is < 0.1% prozent of training and test data

print(train_scale.shape)                                              # look at shapes - same amount features (57)
print(val_scale.shape)
print(test_scale.shape)

train_scale.describe()

"""## Baseline model"""

from sklearn.naive_bayes import GaussianNB                                                              # Naive Bayes
from sklearn.svm import SVC                                                                             # SVM
from sklearn.neighbors import KNeighborsClassifier                                                      # kNN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score                     # evaluation
from sklearn.inspection import permutation_importance                                                   # feature importance (Naive Bayes)

X_train = train_scale.drop('activity',axis=1)                                                           #  raw and transformed time series for X_train
y_train = train_scale['activity']                                                                       # set label for training data
X_val = val_scale.drop('activity', axis=1)                                                              # set label and data for validation set
y_val = val_scale['activity']
X_test = test_scale.drop('activity', axis=1)                                                            # set label and data for test set
y_test = test_scale['activity']


sample_sizetrain=int(len(X_train)*0.5)                                                                 # sample_size for train (X_train or y_train can be used to indicate length)

"""# Naive Bayes"""

modelNB = GaussianNB()                                                  # load model
modelNB.fit(X_train, y_train)                                           # fit model on trainingsdata

predictions = modelNB.predict(X_val)                                    # predict on validation data

accuracy = accuracy_score(y_val, predictions)                           # calculate accuracy
print(f'Accuracy: {accuracy * 100:.2f}%')                               # print results
print(classification_report(y_val, predictions))

"""# feature importance score"""

# found the idea on https://stackoverflow.com/questions/62933365/how-to-get-the-feature-importance-in-gaussian-naive-bayes
imps = permutation_importance(modelNB, X_val, y_val)            #caclulate importance for the model for the training data
print(imps.importances_mean)



"""# kNN"""

modelkNN=KNeighborsClassifier(n_neighbors=3)                    # model
modelkNN.fit(X_train, y_train)                                  # model fit

predictions = modelkNN.predict(X_val)                           # predict

accuracy = accuracy_score(y_val, predictions)                   # calculate accuracy
print(f'Accuracy: {accuracy * 100:.2f}%')                       # print
print(classification_report(y_val, predictions))

#run time onder 4 minutes for accuracy of .94

"""feature importance"""

from sklearn.tree import DecisionTreeClassifier                 # extra package
                                                                # Decision tree for feature ranking
                                                                # Create and fit the model
modelDT = DecisionTreeClassifier(random_state=42)
modelDT.fit(X_train, y_train)
                                                                # Make predictions
predictions = modelDT.predict(X_val)
                                                                # Calculate model accuracy
accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
                                                                # calculate rank features
feature_importances = modelDT.feature_importances_
                                                                # Create a DataFrame
feature_names = X_train.columns if hasattr(X_train, 'columns') else np.arange(X_train.shape[1])
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
                                                                # Print ranked features
print("\nRanked Features:")
print(importance_df)

                                                            # select features
selfea = importance_df['Feature'][:6].tolist()          # create list with 6 feature names
selfea10 = importance_df['Feature'][:10].tolist()       # create list with 10 feature names
print(selfea)
print(selfea10)

"""# SVM on selected features
because runs longer than kNN and NB is not helping in selecting features
"""

modelSVM=SVC(kernel='linear')                                                   #SVM with linear kernel
modelSVM.fit(X_train[selfea][:sample_sizetrain], y_train[sample_sizetrain])

predictions = modelSVM.predict(X_val[selfea])

accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
# sample .1 and running longer than kNN on full set with same variables (9min) was stopped after 545min with 6 features
# (smaple size .1) running time is under 1 minute (60%accuracy)
# running on 10 features (and sample size of .1= was stopped after 185min)

print('classification report: ')
print(classification_report(y_val,predictions))

"""# kNN on selected features"""

modelkNN2=KNeighborsClassifier(n_neighbors=3)                           # kNN slected features 3 neighbors
modelkNN2.fit(X_train[selfea], y_train)

predictions = modelkNN2.predict(X_val[selfea])

accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_val, predictions))

"""1,3,7, neighbors for kNN"""

modelkNN3=KNeighborsClassifier(n_neighbors=1)                       # kNN selected features, neighbor = 1
modelkNN3.fit(X_train[selfea], y_train)

predictions = modelkNN3.predict(X_val[selfea])

accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
#print(classification_report(y_val, predictions))                   # for more details on evaluation

modelkNN4=KNeighborsClassifier(n_neighbors=5)                       # kNN selected features, neighbors = 5
modelkNN4.fit(X_train[selfea], y_train)

predictions = modelkNN4.predict(X_val[selfea])

accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
#print(classification_report(y_val, predictions))                   # for more details on evaluation

modelkNN5=KNeighborsClassifier(n_neighbors=7)                       # kNN selected features, neighbors = 7
modelkNN5.fit(X_train[selfea], y_train)

predictions = modelkNN5.predict(X_val[selfea])

accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
#print(classification_report(y_val, predictions))                   # for more details on evaluation

modelkNN6=KNeighborsClassifier(n_neighbors=10)
modelkNN6.fit(X_train[selfea], y_train)

predictions = modelkNN6.predict(X_val[selfea])

accuracy = accuracy_score(y_val, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
#print(classification_report(y_val, predictions))                   # for more details on evaluation


# best model performance with k = 10 and selected features, running time with 5.6 seconcs fast (94%)

"""# Run on Test data

"""

                                                                    # Naive Bayes
predictions = modelNB.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, predictions))

                                                                    # kNN
predictions = modelkNN.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, predictions))

                                                                    # kNN on selected features (n = 6) n=3
'''modelkNN2=KNeighborsClassifier(n_neighbors=3)
modelkNN2.fit(X_train[selfea], y_train)


predictionskNN2 = modelkNN2.predict(X_test[selfea])

accuracy = accuracy_score(y_test, predictionskNN2)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, predictionskNN2))'''

                                                                    # kNN on selected features (n = 6) (best model)
modelkNN2=KNeighborsClassifier(n_neighbors=10)
modelkNN2.fit(X_train[selfea], y_train)


predictionskNN2 = modelkNN2.predict(X_test[selfea])

accuracy = accuracy_score(y_test, predictionskNN2)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, predictionskNN2))
#78.05 for 3 78,66 for 7, should be tested on the test set.


                                                                    # SVM (trainend on sample n=.5 (? not added to X_test) and selected features n =6)

predictions = modelSVM.predict(X_test[selfea][:sample_size])

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, predictions))

# SVM (trainend on sample .5 and selected features = 10)
'''
modelSVM2=SVC(kernel='linear')
modelSVM2.fit(X_train[selfea10][:sample_size], y_train[:sample_size])

predictions = modelSVM2.predict(X_test[selfea10][:sample_size])

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, predictions))
# >300min'''#>800min>3377min


"""# Visualizing results"""

                                                                        # confusion matrix heatmap

import seaborn as sns

print(classification_report(y_test, predictionskNN2))


print('Confusion Matrix: ')                                             # confusion matrix for kNN2 feature=6 neighbors=10
cm = confusion_matrix(y_test,predictionskNN2)                           # heatmap
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix Heatmap')
plt.show()

"""feature importance

"""

                                                                        # visualize all features
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Feature Importance 56/56')
plt.gca().invert_yaxis()                                                # Invert y-axis to have the highest importance at the top
plt.grid(axis='x')
plt.show()

"""#offener link https://arno.uvt.nl/show.cgi?fid=149145"""

                                                                        # visualize Top 10 features
plt.figure(figsize=(8, 4))
plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10], color='steelblue')
plt.xlabel('Importance', fontsize=14)
plt.title('Feature Importance 10/56', fontsize=16)
plt.gca().invert_yaxis()                                                # Invert y-axis to have the highest importance at the top
plt.grid(axis='x')
plt.show()

