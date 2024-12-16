import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./RawData/rawData.csv')
dataset.head(10)
dataset.shape

dataset.info()

pd.options.display.float_format = "{:.2f}".format
dataset.describe()

dataset[(dataset['GenPrice'].isnull()) | (dataset['GenPrice'] == 0)]

dataset.drop(dataset[(dataset['GenPrice'].isnull()) | (dataset['GenPrice'] == 0)].index, inplace = True )
dataset.shape

dataset[(dataset['PrevAssignedCost'] == 0) & (dataset['AVGCost'] == 0)]

dataset.drop(dataset[(dataset['PrevAssignedCost'] == 0) & (dataset['AVGCost'] == 0)].index, inplace = True)
dataset.shape

dataset[dataset['PrevAssignedCost'] >= 10000000]

dataset.drop(dataset[dataset['PrevAssignedCost'] >= 10000000].index, inplace = True, axis = 0)
dataset.shape

dataset[dataset['GenPrice'] >= 10000000]

dataset.drop(dataset[dataset['GenPrice'] >= 10000000].index, inplace = True, axis = 0)
dataset.shape

def calculateMargin(GenPrice, PrevAssignedCost, AVGCost):
  if PrevAssignedCost > 0.0:
      margin = ((GenPrice - PrevAssignedCost) / PrevAssignedCost) * 100.0
      if margin > 700.0:
          margin = ((GenPrice - AVGCost) / AVGCost) * 100.0
  else:
      margin = ((GenPrice - AVGCost) / AVGCost) * 100.0
  return margin

def calculateMargin(GenPrice, PrevAssignedCost, AVGCost):
  margin = 0
  if PrevAssignedCost > 0:
    margin = ((GenPrice - PrevAssignedCost) / PrevAssignedCost) * 100
  return margin

dataset['Margin'] = dataset.apply(lambda x: calculateMargin(x['GenPrice'], x['PrevAssignedCost'], x['AVGCost']), axis = 1)

dataset['Margin'].describe()

dataset.head(10)

dataset.info()

TypedummyVar = pd.get_dummies(dataset['Type'])
TypedummyVar

dataset2 = pd.concat([dataset[['PrevAVGCost', 'PrevAssignedCost', 'AVGCost', 'LatestDateCost', 'Type', 'GenPrice']], TypedummyVar], axis = 1)
dataset2

dataset2['Cat'] = dataset['SKU'].apply(lambda x : x[0])
dataset2

dataset2['Cat'].unique()

dataset2['SubCat'] = dataset['SKU'].apply(lambda x : x[2])
dataset2

categoryDummy = pd.get_dummies(dataset2['Cat'])
categoryDummy.rename(columns = {'A':'Cat A', 'E':'Cat E'}, inplace = True)
dataset2 = pd.concat([dataset2, categoryDummy], axis = 1)
dataset2

subCategoryDummy = pd.get_dummies(dataset2['SubCat'])
subCategoryDummy.rename(columns = {'A':'SubCat A', 'B':'SubCat B', 'C':'SubCat C', 'D':'SubCat D', 'E':'SubCat E', 'F':'SubCat F', 'G':'SubCat G', 'H':'SubCat H', 'I':'SubCat I', 'J':'SubCat J', 'K':'SubCat K'}, inplace = True)
dataset2 = pd.concat([dataset2, subCategoryDummy], axis = 1)
dataset2

dataset2['Year'] = pd.DatetimeIndex(dataset['date']).year

from pandas.api.types import CategoricalDtype
year_bucket2019 = ['1', '0']
dataset2['Year19'] = pd.cut(dataset2.Year,
                                      bins = [2019, 2020, np.inf],
                                      labels = year_bucket2019,
                                      right = False).astype(str).astype(CategoricalDtype(year_bucket2019, ordered = True))

year_bucket2020 = ['0', '1']
dataset2['Year20'] = pd.cut(dataset2.Year,
                                      bins = [2019, 2020, np.inf],
                                      labels = year_bucket2020,
                                      right = False).astype(str).astype(CategoricalDtype(year_bucket2020, ordered = True))

dataset3 = dataset2.drop(['Type', 'Year', 'Cat', 'SubCat'], axis=1)
dataset3

dataset3.info()

dataset3.to_csv('./CleanedData/cleanData_Final.csv')


