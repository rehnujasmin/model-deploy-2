# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 10:38:05 2021

@author: hp
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
import requests
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

url='https://www.poultrybazaar.net/daily-rate-sheet/Broiler-Rates-Andhra-Pradesh/'
req = requests.get(url)
soup = bs(req.text, 'lxml')

#getting the table:
table = soup.find('table', {'class':'table'})
headers = []

for i in table.find_all('th'):
    title = i.text.strip()
    headers.append(title)

df = pd.DataFrame(columns=headers)
for row in table.find_all('tr')[1:]:
    data = row.find_all('td')
    row_data = [td.text.strip() for td in data]
    length = len(df)
    df.loc[length] = row_data
  
df=df[df['CITY (COMPANY)'].str[0].isin(['h', 'H'])]
df = df.drop('DESC. | DAY >>', 1)
df=df.transpose()
df = df.drop(labels="AVG.", axis=0)
df.columns = df.iloc[0]
df = df[1:]
df['Month']='September'+'-'+'2021'
sr=range(1,32)
df['Day']=sr


k=["2019","2020","2021"]
j = []
import calendar
for o in range(1, 13):
    j.append(calendar.month_name[o]) # month_name is an array


for year in k:
    for months in j:
         url='https://www.poultrybazaar.net/daily-rate-sheet/Broiler-Rates-Andhra-Pradesh/'+year+'/'+months+'/'
         req = requests.get(url)
         soup = bs(req.text, 'lxml')
        
            
            #getting the table:
         table = soup.find('table', {'class':'table'})
         headers = []
            
         for i in table.find_all('th'):
            title = i.text.strip()
            headers.append(title)
            
            df1 = pd.DataFrame(columns=headers)
         for row in table.find_all('tr')[1:]:
            data = row.find_all('td')
            row_data = [td.text.strip() for td in data]
            length = len(df1)
            df1.loc[length] = row_data
            
         df1=df1[df1['CITY (COMPANY)'].str[0].isin(['h', 'H'])]
         df1 = df1.drop('DESC. | DAY >>', 1)
         df1=df1.transpose()
         df1 = df1.drop(labels="AVG.", axis=0)
         df1.columns = df1.iloc[0]
         df1 = df1[1:]
         df1['Month']=months +'-'+ year
         sr=range(1,32)
         df1['Day']=sr 
         df=pd.concat([df, df1], axis=0)


################################ DATA PREPARATION ###########################
nan_value = float("NaN")

df.replace("", nan_value, inplace=True)
df=df.fillna("")
df.columns=['a','b','c','month','day']
df['Hyd_Price'] = df['a']+df['b']+df['c']

final = df.iloc[31:,]
final=final[['day','month','Hyd_Price']]

#reset index

final.reset_index(inplace=True)
final=final.iloc[:1020,]#removing the data from sep 29 onwards


############################## DATA PREPROCESSING ###########################

import re
for i in final.columns:
    final[i][final[i].apply(lambda i: True if re.search('^\s*$', str(i)) else False)]=None

final.isna().sum()#114 na values are present

from sklearn.impute import SimpleImputer

#imputation of null values by mean imputation
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')   
final["Hyd_Price"]=pd.DataFrame(mean_imputer.fit_transform(final[["Hyd_Price"]]))#na values have been imputed by mean    

final.isna().sum() # no na values

from matplotlib import pyplot as plt
plt.boxplot(final['Hyd_Price'])#outliers are present in the data

#outlier treatment
from feature_engine.outliers import Winsorizer
winsorizer=Winsorizer(capping_method="iqr",tail="both",fold=1.5) 
final['Hyd_Price']=winsorizer.fit_transform(pd.DataFrame(final['Hyd_Price'])) 

plt.boxplot(final['Hyd_Price'])#outliers have been removed

cols=["day","month"]
final['date'] = final[cols].apply(lambda x: '-'.join(x.values.astype(str)), axis="columns")
final = final.drop(['day','month'], axis=1)
final = final[~final.date.str.startswith('29-February')]
final = final[~final.date.str.startswith('30-February')]
final = final[~final.date.str.startswith('31-February')]
final = final[~final.date.str.startswith('31-April')]
final = final[~final.date.str.startswith('31-June')]
final = final[~final.date.str.startswith('31-September')]
final = final[~final.date.str.startswith('31-November')]

final['date'] = final['date'].astype('datetime64[ns]')
final.info()  #converted to datetime format

####################### EXPLORATORY DATA ANALYSIS ############################
final.describe()

#first moment business decision
final['Hyd_Price'].mean()#92.8616
final['Hyd_Price'].median()#93
final['Hyd_Price'].mode()#93.8424-- as mean,median&mode are almost equal we conclude the distribution of the data is normal

#second moment business decision
final["Hyd_Price"].std()#19.6623-- standard deviation
final["Hyd_Price"].var()#386.606-- variance

#third moment business decision
final["Hyd_Price"].skew() #0.13756-- skewness(as it is near to zero we consider the data as not skewed i.e., normal)

#fourth moment business decision
final["Hyd_Price"].kurt() #0.14-- kurtosis (normally shaped curved as it is near to zero)

#Data visualization
#barplot
plt.bar(height=final['Hyd_Price'],x=np.arange(1,993,1))
plt.title("Barplot")
#Histogram
plt.hist(final['Hyd_Price'])
plt.title('Histogram')#normal curve.. hence the data is normally distributed.
#boxplot
plt.boxplot(final['Hyd_Price'])
plt.title("Boxplot")

#Normal Quantile-Quantile part
import scipy.stats as stats
import pylab
stats.probplot(final['Hyd_Price'],dist="norm",plot=pylab)#the data is normally distributed

######### MODEL BUILDING ########################

final = final.drop(['index'], axis=1)
final = final[['date', 'Hyd_Price']]

final.set_index('date', inplace=True)
#final.Hyd_Price.mean()
def test_stationarity(ts):
    stats = ['Test Statistic','p-value','Lags','Observations']
    final_test = adfuller(ts, autolag='AIC')
    final_results = pd.Series(final_test[0:4], index=stats)
    for key,value in final_test[4].items():
        final_results['Critical Value (%s)'%key] = value
    print(final_results)

test_stationarity(final['Hyd_Price'])

#train and test split
train = final[:986]
test = final[986:]

### fit ARIMA model
model = ARIMA(train, order=(1,0,0)).fit()
pred = model.predict(start=len(train), end = len(final)-1)
pred

from sklearn.metrics import mean_squared_error
error = np.sqrt(mean_squared_error(test,pred))
error  #11.57

final_model = ARIMA(final, order=(1,0,0)).fit()
prediction= final_model.predict(len(final), len(final)+7)
prediction


#saving model to disk
import pickle
pickle.dump(prediction, open('model.pkl','wb')) #wb - write bytes mode

#loading model to compare results
model = pickle.load(open('model.pkl','rb'))
























