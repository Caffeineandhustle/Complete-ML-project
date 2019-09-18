#!/usr/bin/env python
# coding: utf-8

# In[51]:


#import all the libraries 
#Pandas :  for data analysis
#numpy : for Scientific Computing.
#matplotlib and seaborn : for data visualization
#scikit-learn : ML library for classical ML algorithms
#math :for mathematical functions


import pandas as pd
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_palette('husl')
import warnings
import math
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
#from lightgbm import LGBMClassifier
from sklearn.metrics import  accuracy_score
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
#import lightgbm as  lgb
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler, LabelBinarizer
# auxiliary function
from sklearn.preprocessing import LabelEncoder



# In[5]:


#Read data from csv
iris_data= pd.read_csv('/Users/priyeshkucchu/Desktop/IRIS 2.csv',engine='python')


# In[37]:


#Show top 5 rows

iris_data.head(5)


# In[5]:


#Get detailed information of data 
#checking if there is any inconsistency in the dataset
#as we see there are no null values in the dataset, so the data can be processed
iris_data.info()


# In[8]:


#No of columns in the data

iris_data.shape[1]


# In[9]:


#No of rows in the data

iris_data.shape[0]


# In[41]:


iris_data.shape


# In[10]:


iris_data.species.unique()


# In[11]:


iris_data.tail(5)


# In[14]:


iris_data.count(axis=0)


# In[15]:


iris_data.isnull()


# In[42]:


iris_data.describe()


# In[43]:


iris_data["species"].value_counts()


# In[16]:


iris_data.dropna()


# In[23]:


def missing_values(x):
    return sum(x.isnull())

print("Missing values in each column:")
print(iris_data.apply(missing_values,axis=0))


# In[24]:


type(iris_data)


# # Complete Data Visualization

# In[201]:


import matplotlib.pyplot as plt
import seaborn as sns 

#Read csv

iris_data= pd.read_csv('/Users/priyeshkucchu/Desktop/IRIS 2.csv',engine='python')

#gives us a General Idea about the dataset.
iris_data.describe().plot(kind="area",fontsize=20,figsize=(20,8),table=False, colormap="rainbow")
plt.xlabel("Statistics")
plt.ylabel('Value')
plt.title("Statistics of IRIS dataset")


# In[103]:


#Here the frequency of the observation is plotted.
#In this case we are plotting the frequency of the three species in the Iris Dataset
sns.countplot("species",data=iris_data)
plt.show()


# In[104]:


iris_data["species"].value_counts().plot.pie(explode=[0.1,0.1,0.1], autopct='%1.1f%%',shadow=True,figsize=(10,8))
plt.show()
#We can see that there are 50 samples each of all the Iris Species in the data set.


# In[105]:


#Jointplot is seaborn library specific and can be used to quickly visualize and analyze the relationship between 
#two variables and describe their individual distributions on the same plot.

#Draw a scatterplot with marginal histograms
figure=sns.jointplot(x="sepal_length",y="sepal_width",data=iris_data,color="orange")


# In[106]:


#Replace the scatterplots and histograms with density estimates 

figure=sns.jointplot(x="sepal_length",y="sepal_width",data=iris_data,color="green",kind="kde")


# In[107]:


# Add regression 

figure=sns.jointplot(x="sepal_length",y="sepal_width",data=iris_data,color="blue",kind="reg")


# In[108]:


# Replace the scatterplot with a joint histogram using hexagonal bins

figure=sns.jointplot(x="sepal_length",y="sepal_width",data=iris_data,color="pink",kind="hex")


# In[109]:


#Draw a scatterplot, then add a joint density estimate

figure=(sns.jointplot(x="sepal_length",y="sepal_width",data=iris_data,color="purple").plot_joint(sns.kdeplot,zorder=0,n_levels=6))


# In[202]:


#Facetgrid : Multi-plot grid for plotting conditional relationships.

sns.FacetGrid(iris_data,hue="species",height=7).map(plt.scatter,"sepal_length","sepal_width").add_legend()


# In[111]:


#Boxplot : give a statical summary of the features being plotted.Top line represent the max value,top edge 
#of box is third Quartile, middle edge represents the median,bottom edge represents the first quartile value.
#The bottom most line respresent the minimum value of the feature.The height of the box 
#is called as Interquartile range.The black dots on the plot represent the outlier values in the data.

fig=plt.gcf()
fig.set_size_inches(11,8)
fig=sns.boxplot(x="species", y="petal_width", data=iris_data, hue="species",order=["Iris-setosa","Iris-versicolor","Iris-virginica"],                linewidth=2.5,orient='v',dodge=False )


# In[112]:


#Draw a categorical scatterplot with non-overlapping points.
sns.swarmplot(x="species", y="petal_width", data=iris_data, color=".25")


# In[113]:


#Draw boxplot by species

iris_data.boxplot(by="species",figsize=(10,8))


# In[114]:


#Strip Plot : Draw a scatterplot where one variable is categorical.

#A strip plot can be drawn on its own, but it is also a good complement to 
#a box or violin plot in cases where you want to show all 
#observations along with some representation of the underlying distribution.

fig=plt.gcf()
fig.set_size_inches(11,8)
fig=sns.stripplot(x="species",y="petal_width",data=iris_data,color="blue",hue="species",order=["Iris-setosa",                "Iris-versicolor","Iris-virginica"],jitter=True,edgecolor="black",linewidth=1,size=6,orient='v'                ,palette="Set2")


# In[115]:


#Combine Stripplot and boxplot

fig=plt.gcf()
fig.set_size_inches(11,8)
fig=sns.boxplot(x="species",y="petal_width",data=iris_data)
fig=sns.stripplot(x="species",y="petal_width",data=iris_data,jitter=True, edgecolor="black",hue="species"                 ,linewidth=1.0)


# In[116]:


#Violin Plot It is used to visualize the distribution of data and its probability distribution.
#This chart is a combination of a Box Plot and a Density Plot that is rotated and placed on
#each side, to show the distribution shape of the data. The thick black bar in the centre 
#represents the interquartile range, the thin black line extended from it represents the
#95% confidence intervals, and the white dot is the median.Box Plots are limited in their display of the data, as 
#their visual simplicity tends to hide significant details about how values in the data are distributed.

fig=plt.gcf()
fig.set_size_inches(11,8)
fig=sns.violinplot(x="species",y="petal_width",data=iris_data,hue="species",saturation=0.8,palette="Set3")


# In[117]:


#plot subplot for different columns in the data set

plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
sns.violinplot(x="species",y="sepal_length",data=iris_data,hue="species",saturation=0.8,palette="summer")
plt.subplot(2,2,2)
sns.violinplot(x="species",y="sepal_width",data=iris_data,hue="species",saturation=0.8,palette="summer")
plt.subplot(2,2,3)
sns.violinplot(x="species",y="petal_length",data=iris_data,hue="species",saturation=0.8,palette="summer")
plt.subplot(2,2,4)
sns.violinplot(x="species",y="petal_width",data=iris_data,hue="species",saturation=0.8,palette="summer")



# In[118]:


#Pair Plot: A “pairs plot” is also known as a scatterplot, in which one variable in the same data row 
#is matched with another variable's value. 
#Shows how all variables can be paired with all the other variables.

sns.pairplot(data=iris_data,kind="scatter",hue="species",dropna=True,palette="winter")


# In[119]:


#Heat Map : Heat map is used to find out the correlation between different features in the dataset.High positive or negative value shows that the features have high correlation.
#This helps us to select the parmeters for machine learning.

fig=plt.gcf()
fig.set_size_inches(10,8)
fig=sns.heatmap(iris_data.corr(),vmin=-1,vmax=1,cmap="cubehelix",linewidths=1,linecolor="blue",cbar=True,               cbar_kws={'orientation':'vertical'},square=True,annot=True,mask=False)


# In[126]:


#Distribution plot: The distribution plot is suitable for comparing range and
#distribution for groups of numerical data. Data is plotted as value points along an axis. 
#You can choose to display only the value points to see the distribution of values, a bounding box to
#see the range of values, or a combination of both as shown here.The distribution 
#plot is not relevant for detailed analysis of the data as it deals with a summary of the data distribution.

fig=plt.gcf()
fig.set_size_inches(12,8)
iris_data.hist(bins=10,grid=True,linewidth=1,edgecolor="black")
iris_data.hist(by="species",bins=10,grid=True,linewidth=1,edgecolor="black")


# In[128]:


#LMplot:Plot data and regression model fits across a FacetGrid.
#This function combines regplot() and FacetGrid. 
#It is intended as a convenient interface to fit regression models across conditional subsets of a dataset.

fig=sns.lmplot(x="sepal_length",y="sepal_width",data=iris_data,hue="species",markers='o',palette="winter")


# In[131]:


#FacetGrid

sns.FacetGrid(iris_data,hue="species",height=5)             .map(sns.kdeplot,"sepal_length")             .add_legend()
plt.ioff()


# In[136]:


#Andrews Curve: In data visualization, an Andrews plot or Andrews curve is a way to visualize structure
#in high-dimensional data. It is basically a rolled-down, non-integer version of the Kent–Kiviat 
#radar m chart, or a smoothened version of a parallel coordinate plot.In Pandas
#use Andrews Curves to plot and visualize data structure.Each multivariate observation is 
#transformed into a curve and represents the coefficients of a Fourier 
#series.This useful for detecting outliers in times series data.Use colormap to change the color of the curves

from pandas.plotting import andrews_curves

andrews_curves(iris_data,"species",colormap="rainbow")
plt.show()
plt.ioff()


# In[140]:


#Parallel coordinate plot: This type of visualisation is used for plotting multivariate, 
#numerical data. Parallel Coordinates Plots are ideal for comparing many variables together and 
#seeing the relationships between them. For example, if you had to compare an array of products with 
#the same attributes (comparing computer or cars specs across different models).

from pandas.plotting import parallel_coordinates
parallel_coordinates(iris_data,"species")


# In[143]:


#Radviz Plot : RadViz is a multivariate data visualization algorithm that 
#plots each feature dimension uniformly around the circumference of a 
#circle then plots points on the interior of the circle such that the 
#point normalizes its values on the axes from the center to each arc.

from pandas.plotting import radviz
radviz(iris_data,"species",color=['pink', 'green'])


# In[150]:


#Factorplot: Factor plot is informative when we have multiple groups to compare. 

sns.factorplot("species","sepal_length",data=iris_data)
plt.ioff()
plt.show()


# In[153]:


#Boxen Plot: An enhanced box plot for larger datasets.

fig=plt.gcf()
fig.set_size_inches(10,6)
fig=sns.boxenplot("species","sepal_length",data=iris_data,hue="species",palette="Set3")


# In[157]:


#Residual Plot : The most useful way to plot the residuals, though, is with your predicted values on 
#the x-axis, and your residuals on the y-axis. 
#The distance from the line at 0 is how bad the prediction was for that value.

fig=plt.gcf()
fig.set_size_inches(10,6)
fig=sns.residplot("petal_length","petal_width",data=iris_data,lowess=True,dropna=True,color="blue")


# In[191]:


#Donut Plot : A donut chart is essentially a Pie Chart with an area of the center cut out.

feature_names = "sepal_length","sepal_width","petal_length","petal_width"
feature_size = [len(iris_data["sepal_length"]),len(iris_data["sepal_width"]),len(iris_data                                                                    ["petal_length"]),len(iris_data["petal_width"])]
# create a circle for the center of plot
circle = plt.Circle((0,0),0.2,color = "white")
plt.pie(feature_size, labels = feature_names, colors = ["red","green","blue","cyan"] )
p = plt.gcf()
p.gca().add_artist(circle)
plt.title("Number of Each Features")
plt.show()


# In[198]:


# Create a kde plot of sepal_length versus sepal width for setosa species of flower.
#KDEplot is used to Fit and plot a univariate or bivariate kernel density estimate.

sub=iris_data[iris_data["species"]=="Iris-setosa"]
sns.kdeplot(data=sub[["sepal_length","sepal_width"]],cbar=True,cmap="plasma",shade="True",shade_lowest=False)
plt.title("Iris-Setosa")
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")


# In[4]:


#Venn Diagram : A Venn diagram (also called primary diagram, set diagram or logic diagram) 
#is a diagram that shows all possible logical relations between a finite collection of different sets. 
#Each set is represented by a circle. The circle size represents the importance of the group. 
#The groups are usually overlapping: the size of the overlap represents the intersection between both groups.

from matplotlib_venn import venn2
sepal_length=iris_data.iloc[:,0]
sepal_width=iris_data.iloc[:,1]
petal_length=iris_data.iloc[:,2]
petal_width=iris_data.iloc[:,3]

venn2(subsets=(len(sepal_length)-15,len(sepal_width)-15,15),set_labels=("sepal_length","sepal_width"))
plt.show()


# # Machine Learning

# The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers learn automatically without human intervention or assistance and adjust actions accordingly.

# For that we will split out data set into three parts train, test, validation sets.
# we are going to use the scikit-learn library which has all the required functions and machine learning algorithms required for this notebook
# 
# Before we split our data lets look at the output we want to predict.
# We want to predict the given sepal and petal dimensions follows to which type of species.
# we have 3 type of species Iris-setosa Iris-versicolor Iris-virginica.
# We will convert those species names to a categorical values using label encoding.

# In[6]:


x=iris_data[["sepal_length","sepal_width","petal_length","petal_width"]]
y=iris_data["species"]


# In[7]:


encoder=LabelEncoder()
y= encoder. fit_transform(y)


# In[8]:


y


# In[9]:


#Split the data into train and test set (i.e 70:30 ratio)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)


# Since it is a classification problem we will be using: Logistic Regression

# Logistic regression is a statistical method for analyzing a dataset in which there are 
# one or more independent variables that determine an outcome. 
# The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).

# In[10]:


logist_model=LogisticRegression()
logist_model.fit(x_train,y_train)
prediction=logist_model.predict(x_test)
print("Logistic Regression accuracy : ", accuracy_score(prediction,y_test))


# # SVM
# SVM : “Support Vector Machine” (SVM) is a supervised machine learning algorithm which can be used 
#     for both classification or regression challenges.However, it is mostly used in classification problems. 
#     In this algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you       have) with the value of each feature being the value of a particular coordinate. 
#     Then, we perform classification by finding the hyper-plane that differentiate the two classes very well.
#     Support Vector Machine is a frontier which best segregates the two classes (hyper-plane/ line)

# In[13]:


svm_model= SVC(kernel="linear")
svm_model.fit(x_train,y_train)
svc_prediction=svm_model.predict(x_test)
print("SVM accuracy : ", accuracy_score(svc_prediction,y_test))


# # Naive Bayes
# Naive Bayes : It is a classification technique based on Bayes’ Theorem with an assumption of
#               independence among predictors. In simple terms, a Naive Bayes classifier assumes 
#               that the presence of a particular feature in a class is unrelated to the presence of any other feature.
# 
# For example, a fruit may be considered to be an apple if it is red, round,
# and about 3 inches in diameter. Even if these features depend on each other or 
# upon the existence of the other features, all of these properties independently 
# contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’.

# In[15]:


nb_model= GaussianNB()
nb_model.fit(x_train,y_train)
nb_prediction= nb_model.predict(x_test)
print("Naive Bayes accuracy : ", accuracy_score(nb_prediction,y_test))


# # Decision Trees
# Decision Trees : Decision tree is a type of supervised learning algorithm (having a pre-defined target variable) that
# is mostly used in classification problems. It works for both categorical and continuous input and 
# output variables. In this technique, we split the population or sample into two or more homogeneous
# sets(or sub-populations) based on most significant splitter / differentiator in input variables.

# In[22]:


decision_model=DecisionTreeClassifier(max_leaf_nodes=4)
decision_model.fit(x_train,y_train)
decision_prediction=decision_model.predict(x_test)
print("Decision Tree Accuracy : ", accuracy_score(decision_prediction,y_test))


# # Random Forest 
# Random Forest : Random Forest is a versatile machine learning method capable of performing both 
# regression and classification tasks. It also undertakes dimensional reduction methods, 
# treats missing values, outlier values and other essential steps of data exploration, and does a fairly good job.
# It is a type of ensemble learning method, where a group of weak models combine to form a powerful model.

# In[32]:


random_model=RandomForestClassifier(max_depth=3)
random_model.fit(x_train,y_train)
random_prediction=random_model.predict(x_test)
print("Random Forest Accuracy : ", accuracy_score(random_prediction, y_test))


# # Extra Tree Classifier
# 
# ExtraTreesClassifier is an ensemble learning method fundamentally based on decision trees. ExtraTreesClassifier, like RandomForest, randomizes certain decisions and subsets of data to minimize over-learning from the data and overfitting.Ensembles can give you a boost in accuracy on your dataset.

# In[35]:


extra_model=ExtraTreesClassifier()
extra_model.fit(x_train,y_train)
extra_prediction=extra_model.predict(x_test)
print("Extra tree classifier Accuracy :", accuracy_score(extra_prediction,y_test))


# # KNN
# 
# K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions). KNN has been used in statistical estimation and pattern recognition already in the beginning of 1970's as a non-parametric technique.

# In[47]:


knn_model=KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train,y_train)
knn_prediction=knn_model.predict(x_test)
print("KNN accuracy: ", accuracy_score(knn_prediction,y_test))


# # XGBOOST
# 
# XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.

# In[48]:


xg_model=XGBClassifier()
xg_model.fit(x_train,y_train)
xg_prediction=xg_model.predict(x_test)
print ("XGBoost Accuracy: ", accuracy_score(xg_prediction,y_test))


# In[49]:


cat_model=CatBoostClassifier()
cat_model.fit(x_train,y_train)
cat_prediction=cat_model.predict(x_test)
print("Cat Boost Accuracy : ", accuracy_score(cat_prediction,y_test))


# # Deep Learning

# In[45]:


import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler, LabelBinarizer
X = iris_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_data['species']

X = StandardScaler().fit_transform(X)
y = LabelBinarizer().fit_transform(y)


# In[46]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[47]:


shallow_model = Sequential()
shallow_model.add(Dense( 4, input_dim=4, activation = 'relu'))
shallow_model.add(Dense( units = 10, activation= 'relu'))
shallow_model.add(Dense( units = 3, activation= 'softmax'))
shallow_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[48]:


shallow_history = shallow_model.fit(x_train, y_train, epochs = 150, validation_data = (x_test, y_test))


# In[49]:


plt.plot(shallow_history.history['acc'])
plt.plot(shallow_history.history['val_acc'])
plt.title("Accuracy")
plt.legend(['train', 'test'])
plt.show()


# In[50]:


plt.plot(shallow_history.history['loss'])
plt.plot(shallow_history.history['val_loss'])
plt.plot('Loss')
plt.legend(['Train','Test'])
plt.show()


# # Deep Deep Learning

# In[51]:


deep_model = Sequential()
deep_model.add(Dense( 4, input_dim=4, activation = 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 10, activation= 'relu'))
deep_model.add(Dense( units = 3, activation= 'softmax'))
deep_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[52]:


deep_history = deep_model.fit(x_train, y_train, epochs = 150, validation_data = (x_test, y_test))


# In[53]:


plt.plot(deep_history.history['acc'])
plt.plot(deep_history.history['val_acc'])
plt.title("Accuracy")
plt.legend(['train', 'test'])
plt.show()


# In[54]:


plt.plot(deep_history.history['loss'])
plt.plot(deep_history.history['val_loss'])
plt.plot('Loss')
plt.legend(['Train','Test'])
plt.show()


# In[ ]:




