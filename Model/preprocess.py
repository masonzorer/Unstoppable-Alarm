#testing
import pandas as pd

# open groundtruth labels csv files
train_groundtruth = pd.read_csv('Data/unprocessed/train_groundtruth.csv')
test_groundtruth = pd.read_csv('Data/unprocessed/test_groundtruth.csv')

# drop 'mids' column from groundtruth
train_groundtruth = train_groundtruth.drop(['mids'], axis=1)
test_groundtruth = test_groundtruth.drop(['mids'], axis=1)

# find all rows in groundtruth that have 'Water' in the labels column
train_water = train_groundtruth[train_groundtruth['labels'].str.contains('Water') | 
                              train_groundtruth['labels'].str.contains('water') |
                              train_groundtruth['labels'].str.contains('Liquid') |
                                train_groundtruth['labels'].str.contains('liquid')]
test_water = test_groundtruth[test_groundtruth['labels'].str.contains('Water') |
                            test_groundtruth['labels'].str.contains('water') |
                              test_groundtruth['labels'].str.contains('Liquid') |
                                test_groundtruth['labels'].str.contains('liquid')]

# add column with class 0 to all rows in groundtruth
train_groundtruth['class'] = 0
test_groundtruth['class'] = 0

# make all rows in groundtruth that have the same id as the rows in water have class 1
for index, row in train_water.iterrows():
    train_groundtruth.loc[train_groundtruth['fname'] == row['fname'], 'class'] = 1
for index, row in test_water.iterrows():
    test_groundtruth.loc[test_groundtruth['fname'] == row['fname'], 'class'] = 1

# get all rows in groundtruth that have class 1
train_water = train_groundtruth[train_groundtruth['class'] == 1]
test_water = test_groundtruth[test_groundtruth['class'] == 1]
# get all rows in groundtruth that have class 0
train_no_water = train_groundtruth[train_groundtruth['class'] == 0]
test_no_water = test_groundtruth[test_groundtruth['class'] == 0]

# remove all rows from train_water that have the any of the following in the labels column
# Speech, vehicle, animal, music, engine, walk, traffic
train_water = train_water[~train_water['labels'].str.contains('Vehicle') &
                            ~train_water['labels'].str.contains('vehicle') &
                            ~train_water['labels'].str.contains('Animal') &
                            ~train_water['labels'].str.contains('animal') &
                            ~train_water['labels'].str.contains('Music') &
                            ~train_water['labels'].str.contains('music') &
                            ~train_water['labels'].str.contains('Engine') &
                            ~train_water['labels'].str.contains('engine') &
                            ~train_water['labels'].str.contains('Walk') &
                            ~train_water['labels'].str.contains('walk') &
                            ~train_water['labels'].str.contains('Traffic') &
                            ~train_water['labels'].str.contains('traffic') &
                            ~train_water['labels'].str.contains('Speech') &
                            ~train_water['labels'].str.contains('speech')]
test_water = test_water[~test_water['labels'].str.contains('Vehicle') &
                            ~test_water['labels'].str.contains('vehicle') &
                            ~test_water['labels'].str.contains('Animal') &
                            ~test_water['labels'].str.contains('animal') &
                            ~test_water['labels'].str.contains('Music') &
                            ~test_water['labels'].str.contains('music') &
                            ~test_water['labels'].str.contains('Engine') &
                            ~test_water['labels'].str.contains('engine') &
                            ~test_water['labels'].str.contains('Walk') &
                            ~test_water['labels'].str.contains('walk') &
                            ~test_water['labels'].str.contains('Traffic') &
                            ~test_water['labels'].str.contains('traffic') &
                            ~test_water['labels'].str.contains('Speech') &
                            ~test_water['labels'].str.contains('speech')]

# recombine class 1 and class 0 rows in groundtruth
train_groundtruth = pd.concat([train_water, train_no_water], ignore_index=True)
test_groundtruth = pd.concat([test_water, test_no_water], ignore_index=True)

# rename 'fname' column to 'id' in groundtruth
train_groundtruth = train_groundtruth.rename(columns={'fname': 'id'})
test_groundtruth = test_groundtruth.rename(columns={'fname': 'id'})

# split groundtruth into train and dev based on split column
dev_groundtruth = train_groundtruth[train_groundtruth['split'] == 'val']
train_groundtruth = train_groundtruth[train_groundtruth['split'] == 'train']

# drop split column from train and dev groundtruth
train_groundtruth = train_groundtruth.drop(['split'], axis=1)
dev_groundtruth = dev_groundtruth.drop(['split'], axis=1)

# drop 'labels' column from all groundtruth
train_groundtruth = train_groundtruth.drop(['labels'], axis=1)
dev_groundtruth = dev_groundtruth.drop(['labels'], axis=1)
test_groundtruth = test_groundtruth.drop(['labels'], axis=1)

# get all rows in train that have class 1
train_water = train_groundtruth[train_groundtruth['class'] == 1]
# oversample class 1 in train by creating 3 copies of each row
train_water = pd.concat([train_water]*3, ignore_index=True)

# get all rows in train that have class 0
train_no_water = train_groundtruth[train_groundtruth['class'] == 0]
# randomly undersample class 0 in dev
train_no_water = train_no_water.sample(n=15000)

# combine class 1 and class 0 rows in train
train_groundtruth = pd.concat([train_water, train_no_water], ignore_index=True)
train_groundtruth = train_groundtruth.sample(frac=1).reset_index(drop=True)

# remove 3/4 of the rows in dev that have class 0
dev_water = dev_groundtruth[dev_groundtruth['class'] == 1]
dev_no_water = dev_groundtruth[dev_groundtruth['class'] == 0]
dev_no_water = dev_no_water.sample(frac=0.25)
dev_groundtruth = pd.concat([dev_water, dev_no_water], ignore_index=True)
dev_groundtruth = dev_groundtruth.sample(frac=1).reset_index(drop=True)

# print shape of all data
print('Train data shape: {}'.format(train_groundtruth.shape))
print('Dev data shape: {}'.format(dev_groundtruth.shape))
print('Test data shape: {}'.format(test_groundtruth.shape))

# print value counts of classes in all data
print(train_groundtruth['class'].value_counts())
print(dev_groundtruth['class'].value_counts())
print(test_groundtruth['class'].value_counts())

# save all data to csv files
train_groundtruth.to_csv('Data/train_groundtruth.csv', index=False)
dev_groundtruth.to_csv('Data/dev_groundtruth.csv', index=False)
test_groundtruth.to_csv('Data/test_groundtruth.csv', index=False)








