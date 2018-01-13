import pyspark.sql as sq
from pyspark.sql.functions import *
import os
import numpy as np

spark = sq.SparkSession.builder.master("local").appName("my app").config("spark.some.config.option", "some-value").getOrCreate()

#path="/home/iman/Dropbox/work-samples-master/data-mr/data"
path="/tmp/data"

poi = spark.read.csv(os.path.join(path,"POIList.csv"),header=True,inferSchema = True)
data = spark.read.csv(os.path.join(path,"DataSample.csv"),header=True,inferSchema = True)
#***************************************
#***********  1.CLEANUP  ***************
#***************************************
#Finding the suspicous IDs
aa=(data.groupBy([' TimeSt', 'Latitude', 'Longitude']).agg(collect_list("_ID").alias("_ID2")).where(size("_ID2") > 1)).select(explode("_ID2").alias("_ID"))
data=data.join(aa, data._ID == aa._ID, "left_anti").drop(aa._ID) #Removing the suspicous IDs
#***************************************
#***********  2.LABEL  *****************
#***************************************
#defining a function for calculating distance between two points
def dis(lat1,lon1,lat2,lon2):
    R=6371
    lon1=toRadians(lon1)
    lat1=toRadians(lat1)
    lon2=toRadians(lon2)
    lat2=toRadians(lat2)
    a=sin((lat1-lat2)/2)**2+cos(lat1)*cos(lat2)*sin((lon1-lon2)/2)**2
    c=2*atan2(sqrt(a),sqrt(1-a))
    d=R*c
    return d

poi = poi.select('POIID',col(" Latitude").alias("poiLatitude"), col("Longitude").alias("poiLongitude"))
data=data.crossJoin(poi)
data=data.withColumn("Poidistance", dis(data['Latitude'],data['Longitude'], data['poiLatitude'],data['poiLongitude']))

data2=data.groupBy('_ID').min('Poidistance')
data=data.join(data2,(data['_ID'] == data2['_ID']) & (data['Poidistance'] == data2['min(Poidistance)'])).drop(data2._ID)
data=data.drop('Poidistance')
#***************************************
#*************  3.ANALYSIS  ************
#***************************************
Q1=data.approxQuantile("min(Poidistance)", [0.25], 0.05)
Q3=data.approxQuantile("min(Poidistance)", [0.75], 0.05)
IQR = Q3[0] - Q1[0]
lowerRange = Q1[0] - 1.5*IQR
upperRange = Q3[0] + 1.5*IQR

#Average and Standard deviation with outliers
analysis1=data.groupBy('POIID').agg(avg("min(Poidistance)").alias('AVG(with outliers)'), stddev("min(Poidistance)").alias('STD(with outliers)'))
#analysis1.show()

#Average and Standard deviation without outliers
analysis12=data[(data['min(Poidistance)']>lowerRange) & (data['min(Poidistance)']<upperRange)].groupBy('POIID').agg(avg("min(Poidistance)").alias('AVG(without outliers)'), stddev("min(Poidistance)").alias('STD(without outliers)'))
#analysis12.show()

#Radius and Density with outliers
analysis2=data.groupBy('POIID').agg(max("min(Poidistance)").alias('Radius(with outliers)'), count("min(Poidistance)").alias('Count(with outliers)'))
analysis2=analysis2.withColumn('density(with outliers)',analysis2['Count(with outliers)']/(analysis2['Radius(with outliers)']**2*np.pi))
#analysis2.show()

#Radius and Density without outliers
analysis22=data[(data['min(Poidistance)']>lowerRange) & (data['min(Poidistance)']<upperRange)].groupBy('POIID').agg(max("min(Poidistance)").alias('Radius(without outliers)'), count("min(Poidistance)").alias('Count(without outliers)'))
analysis22=analysis22.withColumn('density(without outliers)',analysis22['Count(without outliers)']/(analysis22['Radius(without outliers)']**2*np.pi))
#analysis22.show()
#***************************************
#************  4.Model  ****************
#***************************************
#To be more sensetive around mean average, I changed the formula of Q1 and Q3 to 0.35 and 0.65 respectivrely.
Q1=data.approxQuantile("min(Poidistance)", [0.35], 0.05)
Q3=data.approxQuantile("min(Poidistance)", [0.65], 0.05)
IQR = Q3[0] - Q1[0]
lowerRange = Q1[0] - 1.5*IQR
upperRange = Q3[0] + 1.5*IQR

analysis3=data[(data['min(Poidistance)']>lowerRange) & (data['min(Poidistance)']<upperRange)].groupBy('POIID').agg(max("min(Poidistance)").alias('Radius(without outliers)'), count("min(Poidistance)").alias('Count(without outliers)'))
analysis3=analysis3.withColumn('density(without outliers)',analysis3['Count(without outliers)']/(analysis3['Radius(without outliers)']**2*np.pi))

#MODEL: f(x)=(b-a)*(x-min)/(max-min)+a; b=10, a=-10
analysis31=analysis3.toPandas() 
minn=np.min(analysis31['density(without outliers)'])
maxx=np.max(analysis31['density(without outliers)'])
analysis31['density(mapped)']=20*(analysis31['density(without outliers)']-minn)/(maxx-minn)-10
analysis31
#***************************************
#********* BONUS ***********************

#Please Uncomment the following lines if you want to check the results in a non-spark medium

#import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.datasets import make_classification
#from sklearn.ensemble import ExtraTreesClassifier
#bonus=data.toPandas()
#columns = ['_ID', 'poiLongitude', 'poiLatitude']
#bonus.drop(columns, inplace=True, axis=1)
#bonus.dtypes
#bonus['hour']=bonus[' TimeSt'].dt.hour
#bonus['day']=bonus[' TimeSt'].dt.day
#bonus['month']=bonus[' TimeSt'].dt.month
#bonus.describe()
#
#forest = ExtraTreesClassifier(n_estimators=50,random_state=0)
#y=bonus.iloc[:,6]
#X=bonus.iloc[:,0:9]
#X.drop('POIID',inplace=True, axis=1)
#X.drop(' TimeSt',inplace=True, axis=1)
#X2=pd.get_dummies(X)
#forest.fit(X2, y)
#
#features = X2.columns
#importances = forest.feature_importances_
#indices = np.argsort(importances)
#imp=pd.DataFrame()
#imp['features']=features[indices]
#imp['Score']=importances[indices]
#imp.sort_values('Score', ascending=False)


