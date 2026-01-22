import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

#1. Load the dataset

housing=pd.read_csv("housing.csv")

#2. Create a statified a test set

housing["income_cat"]=pd.cut(housing["median_income"],
                             bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
                             labels=[1,2,3,4,5])

split=StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)

for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set=housing.loc[train_index].drop("income_cat",axis=1)# we will work on that 
    strat_test_set=housing.loc[test_index].drop("income_cat",axis=1) #test aside the data

# We will work on the copy of trainig dataset 
housing=strat_train_set.copy()

#3. Seprate features and labels 
housing_labels=housing["median_house_value"].copy()
housing=housing.drop("median_house_value",axis=1)

#print(housing,housing_labels)

#4. List numerical and categorical columns

num_attribute=housing.drop("ocean_proximity",axis=1).columns.tolist()
cat_attribute=["ocean_proximity"]

#5.Lets make the pipeline for numerical columns

#For numerical
num_pipeline=Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

#For categorical
cat_pipline=Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


#Construct full pipeline
full_pipeline=ColumnTransformer([
    ("num",num_pipeline,num_attribute),
    ("cat",cat_pipline,cat_attribute)
])

#6. Transfrom the data 
housing_prepare=full_pipeline.fit_transform(housing)
print(housing_prepare.shape)

#7. Train the model

#linear regression model
lin_reg=LinearRegression()
lin_reg.fit(housing_prepare,housing_labels)
lin_preds=lin_reg.predict(housing_prepare)
#lin_rmse=root_mean_squared_error(housing_labels,lin_preds,)
#print(f"The root means squared error for linear regression is {lin_rmse}")
lin_rmses= -cross_val_score(lin_reg, housing_prepare, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(lin_rmses).describe())


#Desicion tree model
dec_reg=DecisionTreeRegressor()
dec_reg.fit(housing_prepare,housing_labels)
dec_preds=dec_reg.predict(housing_prepare)
#dec_rmse=root_mean_squared_error(housing_labels,dec_preds,)
#print(f"The root means squared error for Desicion Tree is {dec_rmses}")
dec_rmses= -cross_val_score(dec_reg, housing_prepare, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(dec_rmses).describe())


#Random forest model
ran_reg=RandomForestRegressor()
ran_reg.fit(housing_prepare,housing_labels)
ran_preds=ran_reg.predict(housing_prepare)
#ran_rmse=root_mean_squared_error(housing_labels,ran_preds,)
#print(f"The root means squared error for Random forest is {ran_rmse}")

ran_rmses= -cross_val_score(ran_reg, housing_prepare, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(ran_rmses).describe())