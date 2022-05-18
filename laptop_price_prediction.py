# In[0]: IMPORT AND FUNCTIONS
#region
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from sklearn.utils import column_or_1d 
import regex as re
from statistics import mean
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder 
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
#endregion

# In[1]: PART 1. LOOK AT THE BIG PICTURE (DONE)

# In[2]: PART 2. GET THE DATA (DONE). LOAD DATA
raw_data = pd.read_csv('datasets\laptop_price.csv', encoding='ISO-8859-1')

# In[3]: PART 3. DISCOVER THE DATA TO GAIN INSIGHTS
# 3.1 Quick view of the data
print('\n____________ Dataset info ____________')
print(raw_data.info())     
print('\n____________ Some first data examples ____________')
print(raw_data.head()) 
print('\n____________ Statistics of numeric features ____________')
print(raw_data.describe()) 

# 3.2 Scatter plot b/w 2 features
raw_data.plot(kind="scatter", y="Price_euros", x="TypeName", alpha=0.2)
plt.savefig('figures/scatter_laptop_feat.png', format='png', dpi=300)
plt.show()

# 3.4 Plot histogram of 1 feature
from pandas.plotting import scatter_matrix   
features_to_plot = ["Price_euros"]
scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
plt.show()

# 3.5 Plot histogram of numeric features
raw_data.hist(figsize=(10,5)) #bins: no. of intervals
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.tight_layout()
plt.savefig('figures/hist_raw_data.png', format='png', dpi=300) # must save before show()
plt.show()

# In[4]: 
#region
# Đổi tên cột thành chữ thường
raw_data = raw_data.rename(columns=str.lower)

# Đổi tên cột Price_euros thành Price
raw_data = raw_data.rename(columns={'price_euros':'price'})

# Remove unused feature: laptop_ID
raw_data = raw_data.drop('laptop_id', axis=1)

# In thông tin ScreenResolution
raw_data['screenresolution']
# Đếm số lượng 
raw_data['screenresolution'].value_counts()

# Chia ScreenResolution thành 3 cột
raw_data['resolution'] = raw_data['screenresolution'].str.extract(r'(\d+x\d+)')
raw_data['resolution'].value_counts()

raw_data['screen_type'] = raw_data['screenresolution'].replace(r'(\d+x\d+)', '', regex=True)

raw_data['screen_type'] = raw_data['screen_type'].replace(r'(Full HD|Quad HD|Quad HD|\+|/|4K Ultra HD)', '', regex=True)
raw_data['resolution'].value_counts()

# Chia screen type thành 2 loại có hoặc không có touch screen
raw_data['touch_screen'] = raw_data['screen_type'].str.extract(r'(Touchscreen)')
raw_data['screen_type'] = raw_data['screen_type'].str.replace(r'(Touchscreen)', '', regex=True)
raw_data['screen_type'] = raw_data['screen_type'].replace(r' ', '', regex=True)

raw_data['screen_type'].value_counts()
raw_data['screen_type'] = raw_data['screen_type'].replace(r'^\s*$', np.nan, regex=True)
raw_data['screen_type'].value_counts()

raw_data['touch_screen'].value_counts()

raw_data['touch_screen'] = raw_data['touch_screen'].replace('Touchscreen', 1)
raw_data['touch_screen'] = raw_data['touch_screen'].replace(np.nan, 0)

raw_data['touch_screen'].value_counts()

raw_data = raw_data.drop('screenresolution', axis=1)
raw_data.head()


# Feature CPU
raw_data['cpu']

# Cắt tần số CPU ra thành một feature riêng
raw_data['cpu_freq'] = raw_data['cpu'].str.extract(r'(\d+(?:\.\d+)?GHz)')
raw_data['cpu_freq'] = raw_data['cpu_freq'].str.replace('GHz', '')
raw_data.rename(columns={'cpu_freq':'cpu_freq(GHz)'}, inplace=True)

raw_data['cpu_freq(GHz)'].value_counts()

# Giữ lại các thông tin chung
raw_data['cpu'] = raw_data['cpu'].str.replace(r'^Intel Core i5.*GHz$', 'Intel Core i5', regex=True)
raw_data['cpu'] = raw_data['cpu'].str.replace(r'^Intel Core i7.*GHz$', 'Intel Core i7', regex=True)
raw_data['cpu'] = raw_data['cpu'].str.replace(r'^Intel Core i3.*GHz$', 'Intel Core i3', regex=True)
raw_data['cpu'] = raw_data['cpu'].str.replace(r'^AMD.*GHz$', 'AMD', regex=True)
raw_data['cpu'] = raw_data['cpu'].str.replace(r'^Samsung.*GHz$', 'Samsung', regex=True)
raw_data['cpu'] = raw_data['cpu'].str.replace(r'(^(?!(Intel Core i3)|(Intel Core i5)|(Intel Core i7)|(AMD)|(Samsung)).*$)', 'Intel other', regex=True)
raw_data['cpu'].value_counts()

# CPU brand
raw_data['cpu_brand'] = raw_data['cpu'].str.extract(r'^(\w+)')
raw_data['cpu_brand'].value_counts()

# Loại CPU samsung vì số lượng quá ít (1 samples)
raw_data[raw_data['cpu_brand'] == 'Samsung']
raw_data = raw_data.drop(1191)

# Feature RAM
raw_data['ram'] = raw_data['ram'].str.replace('GB', '')
raw_data.rename(columns={'ram': 'ram(GB)'}, inplace=True)
raw_data['ram(GB)'] = raw_data['ram(GB)'].astype(float)
raw_data.head()

# Feature Memory
raw_data['memory']
raw_data['memory'].value_counts()

# Chuyển đơn vị về GB
raw_data['memory_1st'] = raw_data['memory']
raw_data['memory_1st'] = raw_data['memory_1st'].str.replace('1.0TB', '1TB', regex=True)
raw_data['memory_1st'] = raw_data['memory_1st'].str.replace('1TB', '1000GB')
raw_data['memory_1st'] = raw_data['memory_1st'].str.replace('2TB', '2000GB')

# Xóa đơn vị GB
raw_data['memory_1st'] = raw_data['memory_1st'].str.replace('GB', '')
raw_data['memory_1st'].value_counts()

raw_data['memory_2nd'] = raw_data['memory_1st'].str.replace(r' ', '')
raw_data['memory_2nd'].value_counts()

memory_1st, memory_2nd = [], []

for i in raw_data['memory_2nd']:
    if (len(re.findall(r'\+', i)) == 1):
        mem = re.findall(r'(\w+)', i)
        memory_1st.append(mem[0])
        memory_2nd.append(mem[1])
    else:
        mem = re.findall(r'(\w+)', i)
        memory_1st.append(mem[0])
        memory_2nd.append('NaN')

memory_1st_gb, memory_1st_type = [], []
for i in memory_1st:
    memory_1st_type.append(re.findall(r'(\D\w+)', i)[0])
    memory_1st_gb.append(re.findall(r'(\d+)', i)[0])

memory_2nd_gb, memory_2nd_type = [], []
for i in memory_2nd:
    if i == 'NaN':
        memory_2nd_type.append('NaN')
        memory_2nd_gb.append(0)
    else:
        memory_2nd_type.append(re.findall(r'(\D\w+)', i)[0])
        memory_2nd_gb.append(re.findall(r'(\d+)', i)[0])

raw_data['memory_1st_type'] = memory_1st_type
raw_data['memory_1st_sto(GB)'] = memory_1st_gb
raw_data['memory_2nd_type'] = memory_2nd_type
raw_data['memory_2nd_sto(GB)'] = memory_2nd_gb

raw_data['memory_1st_sto(GB)'] = raw_data['memory_1st_sto(GB)'].astype(float)
raw_data['memory_2nd_sto(GB)'] = raw_data['memory_2nd_sto(GB)'].astype(float)

raw_data = raw_data.drop(['memory', 'memory_1st', 'memory_2nd'], axis=1)
raw_data.replace({'NaN': np.nan})
raw_data.head()

# Feature Weight
raw_data['weight'].value_counts()
raw_data['weight'] = raw_data['weight'].str.replace('kg', '').astype(float)
raw_data.rename(columns={'weight': 'weight(kg)'}, inplace=True)
raw_data.head()

# Feature GPU
# Tách GPU brand thành 1 feature
raw_data['gpu_brand'] = raw_data['gpu'].str.extract(r'^(\w+)')
raw_data['gpu_brand'].value_counts()

raw_data['gpu'].value_counts()

# Lưu dữ liệu sau khi clean vào file
raw_data.to_csv('datasets/clean_latop_data.csv', index=False)

#%% Khám phá lại dữ liệu sau khi chuyển đổi
clean_data = pd.read_csv('datasets/clean_latop_data.csv')


print('\n____________ Dataset info ____________')
print(clean_data.info())              
print('\n____________ Some first data examples ____________')
print(clean_data.head())
print('\n____________ Counts on a feature ____________')
print(clean_data['cpu'].value_counts()) 
print('\n____________ Statistics of numeric features ____________')
print(clean_data.describe())   


#%% Plot 
clean_data.plot(kind="scatter", y="price", x="weight(kg)", alpha=0.2)
plt.savefig('figures/scatter_laptop_feat.png', format='png', dpi=300)
plt.show() 

clean_data.plot(kind="scatter", y="price", x="gpu_brand", alpha=0.2)
plt.show()

clean_data.plot(kind="scatter", y="price", x="ram(GB)", alpha=0.2)
plt.show()

from pandas.plotting import scatter_matrix   
features_to_plot = ["price", "weight(kg)", "inches", "ram(GB)"]
scatter_matrix(clean_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
#plt.savefig('figures/scatter_mat_all_feat.png', format='png', dpi=300)
plt.show()

features_to_plot = ["weight(kg)"]
scatter_matrix(clean_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
plt.show()

features_to_plot = ["ram(GB)"]
scatter_matrix(clean_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
plt.show()

clean_data.hist(figsize=(10,5)) #bins: no. of intervals
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.tight_layout()
plt.savefig('figures/hist_clean_data.png', format='png', dpi=300) # must save before show()
plt.show()

#%% Correlation
corr_matrix = clean_data.corr()
print(corr_matrix) # print correlation matrix
print('\n',corr_matrix["price"].sort_values(ascending=False)) 

#In[5]: PART 4. PREPARE THE DATA 
# Xóa cột product vì không quyết định đến price, cột gpu vì đã có cột gpu_brand tổng quát hơn
clean_data = clean_data.drop('product', axis=1)
clean_data = clean_data.drop('gpu', axis=1)

#%% One hot encoding
# clean_data = pd.get_dummies(clean_data, columns=cat_feature_names, drop_first=True)

#%% Split training-test set
train_set, test_set = train_test_split(clean_data, test_size=0.2, random_state=42)

print('\n____________ Split training and test set ____________')     
print(len(train_set), "training +", len(test_set), "test examples")
print(train_set.head())

#%% Separate labels from data
train_set_labels = train_set["price"].copy()
train_set = train_set.drop(columns = "price") 
test_set_labels = test_set["price"].copy()
test_set = test_set.drop(columns = "price") 

#%% Define pipelines
# Define ColumnSelector: a transformer for choosing columns
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values 


# cat_feature_names = ['company', 'product', 'typename', 'cpu', 'gpu', 'opsys', 'resolution',
#     'screen_type', 'memory_1st_type', 'memory_2nd_type', 'cpu_brand', 'gpu_brand']
num_feature_names = ['inches', 'ram(GB)', 'weight(kg)', 'touch_screen', 'cpu_freq(GHz)',
    'memory_1st_sto(GB)', 'memory_2nd_sto(GB)']
cat_feature_names = list(train_set.select_dtypes(exclude=[np.number]))
#%% Pipeline for categorical features
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feature_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place 
    ('cat_encoder', OneHotEncoder()) # convert categorical data into one-hot vectors
])  

# %% Pipeline for numerical features
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feature_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)),
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) # Scale features to zero mean and unit variance
])  

#%% Combine features transformed by two above pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])  

#%% Run the pipeline to process training data
processed_train_set = full_pipeline.fit_transform(train_set)
print('\n____________ Processed feature values ____________')
print(processed_train_set.shape)
print('We have {num} numeric feature and + {cols} cols of onehotvector for categorical features.'.format(num = len(num_feature_names), cols = processed_train_set.shape[1] - len(num_feature_names)))
save_to_file = True
print(processed_train_set[[0, 1, 2],:].toarray())
print(processed_train_set.shape)
if save_to_file:
    joblib.dump(full_pipeline, r'models/full_pipeline.pkl')


# In[6]: PART 5. TRAIN AND EVALUATE MODELS 

# Compute R2 score and root mean squared error
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse 

# Store models to files, to compare latter
def store_model(model, model_name = ""):
    # NOTE: sklearn.joblib faster than pickle of Python
    # INFO: can store only ONE object in a file
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'models/' + model_name + '_model.pkl')
def load_model(model_name):
    # Load objects into memory
    #del model
    model = joblib.load('models/' + model_name + '_model.pkl')
    #print(model)
    return model

#%% Try Linear regression
linear_reg = LinearRegression()
linear_reg.fit(processed_train_set, train_set_labels)
r2score, rmse = r2score_and_rmse(linear_reg, processed_train_set, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))


#%% Ridge
elastic_net = ElasticNet(l1_ratio=0.9 ,random_state=42)
elastic_net.fit(processed_train_set, train_set_labels)
r2score, rmse = r2score_and_rmse(elastic_net, processed_train_set, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
#%% SVM
svm_reg = SVR(kernel='rbf', degree=2, epsilon=0.1, C=100, gamma="scale")
svm_reg.fit(processed_train_set, train_set_labels)
r2score, rmse = r2score_and_rmse(svm_reg, processed_train_set, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

#%%
knn_reg = KNeighborsRegressor(n_neighbors=10, weights='uniform')
knn_reg.fit(processed_train_set, train_set_labels)
r2score, rmse = r2score_and_rmse(knn_reg, processed_train_set, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

#%% Try RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=5)
rf_reg.fit(processed_train_set, train_set_labels)
print('\n____________ RandomForestRegressor ____________')
r2score, rmse = r2score_and_rmse(rf_reg, processed_train_set, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(rf_reg)

print("\nPredictions: ", rf_reg.predict(processed_train_set[0:9]).round(decimals=1))
print("Labels:      ", list(train_set_labels[0:9]))

#%% Evaluate with K-fold cross validation 
# NOTE: 
#   + If data labels are float, cross_val_score use KFold() to split cv data.
#   + KFold randomly splits data, hence does NOT ensure data splits are the same (only StratifiedKFold may ensure that)
#%% 
model = LinearRegression()     
nmse_scores = cross_val_score(model, processed_train_set, train_set_labels, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-nmse_scores)
joblib.dump(rmse_scores,'saved_objects/' + 'LinearRegression' + '_rmse.pkl')
print("LinearRegression rmse: ", rmse_scores.round(decimals=1))
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')


#%%
model = SVR(kernel='rbf', degree=2, epsilon=0.1, C=100, gamma="scale")     
nmse_scores = cross_val_score(model, processed_train_set, train_set_labels, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-nmse_scores)
joblib.dump(rmse_scores,'saved_objects/' + 'SVR' + '_rmse.pkl')
print("SVR rmse: ", rmse_scores.round(decimals=1))
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

#%%
model = ElasticNet(l1_ratio=0.9, random_state=42)          
nmse_scores = cross_val_score(model, processed_train_set, train_set_labels, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-nmse_scores)
joblib.dump(rmse_scores,'saved_objects/' + 'ElasticNet' + '_rmse.pkl')
print("ElasticNet rmse: ", rmse_scores.round(decimals=1))
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')



#%% KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=10, weights='uniform')     
nmse_scores = cross_val_score(model, processed_train_set, train_set_labels, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-nmse_scores)
joblib.dump(rmse_scores,'saved_objects/' + 'KNeighborsRegressor' + '_rmse.pkl')
print("KNeighborsRegressor rmse: ", rmse_scores.round(decimals=1))
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

#%%
model = RandomForestRegressor(n_estimators=5)             
nmse_scores = cross_val_score(model, processed_train_set, train_set_labels, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-nmse_scores)
joblib.dump(rmse_scores,'saved_objects/' + 'RandomForestRegressor' + '_rmse.pkl')
print("RandomForestRegressor rmse: ", rmse_scores.round(decimals=1))
print("Avg. rmse: ", mean(rmse_scores.round(decimals=1)),'\n')

# In[7]: PART 6. FINE-TUNE MODELS 
print('\n____________ Fine-tune models ____________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    #print('Best estimator: ', grid_search.best_estimator_) # NOTE: require refit=True in  SearchCV
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params) 

method = 2
if method == 1:
    from sklearn.model_selection import GridSearchCV
        
    run_new_search = 1      
    if run_new_search:        
        # 6.1.1 Fine-tune RandomForestRegressor
        model = RandomForestRegressor()
        bootstrap = [True, False]
        n_estimators = [20, 50, 100]
        max_features = ['auto', 'sqrt', 'log2']
        max_depth = [15, 20, 30]
        param_grid = {
            'bootstrap': bootstrap,
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth
        }
            # Train across 5 folds, hence a total of (15+12)*5=135 rounds of training 
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, 
        refit=True, verbose=3) # refit=True: after finding best hyperparam, it fit() the model with whole data (hope to get better result)
        grid_search.fit(processed_train_set, train_set_labels)
        joblib.dump(grid_search,'saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")  


        # Fine-tune SVM
        model = SVR()
        C = [1, 100, 1000, 2000, 4000, 5000, 10000, 40000]
        degree = [1, 2]
        epsilon = [1, 5, 10, 20, 30, 40]
        param_grid = {
            'C': C,
            'epsilon': epsilon,
            'degree': degree
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, 
        refit=True, verbose=3, n_jobs=5)
        grid_search.fit(processed_train_set, train_set_labels)
        joblib.dump(grid_search,'saved_objects/SVR_gridsearch.pkl')
        print_search_result(grid_search, model_name = "SVR")  

        # Fine-tune ElasticNet
        model = ElasticNet()
        alpha = [0.1, 0.2, 0.5, 0.7, 0.9, 1, 10]
        l1_ratio = [0.2, 0.5, 0.7, 0.9, 1]
        max_iter = [500, 1000, 2000, 4000]
        random_state = [42]
        param_grid = {
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'max_iter': max_iter,
            'random_state': random_state
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, 
        refit=True, verbose=3)
        grid_search.fit(processed_train_set, train_set_labels)
        joblib.dump(grid_search,'saved_objects/ElasticNet_gridsearch.pkl')
        print_search_result(grid_search, model_name = "ElasticNet")

        # Fine-tune KNeighborsRegressor
        model = KNeighborsRegressor()
        n_neighbors = [3, 5, 10, 15, 20, 30]
        weights = ['uniform', 'distance']
        param_grid = {
            'n_neighbors': n_neighbors,
            'weights': weights,
            'algorithm': algorithms
        }
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, 
        refit=True, verbose=3)
        grid_search.fit(processed_train_set, train_set_labels)
        joblib.dump(grid_search,'saved_objects/KNeighborsRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "KNeighborsRegressor")
        
    else:
        # Load grid_search
        grid_search = joblib.load('saved_objects/RandomForestRegressor_gridsearch.pkl')
        print_search_result(grid_search, model_name = "RandomForestRegressor")         
else:
    from sklearn.model_selection import RandomizedSearchCV
    cv = KFold(n_splits=5,shuffle=True,random_state=42)
    run_new_search = 1      
    if run_new_search: 
        model = RandomForestRegressor()

        bootstrap = [True, False]    # có bootstrap hay không
        max_depth = [15, 20, 30]
        # số lượng Tree của RandomForest 
        # sử dụng numpy.linspace để lấy ra 10 số trong khoảng từ 3 đến 50
        n_estimators = [int(x) for x in np.linspace(start=10, stop=120, num=10)]      
        max_features = [0.2, 0.5, 0.7, 1.0]

        distributions = {
            'bootstrap': bootstrap,
            'max_features': max_features,
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }

        # RandomizedSearchCV với n_iter = 20
        randomized_search = RandomizedSearchCV(model, distributions, n_iter=20,
            cv=cv, scoring='neg_mean_squared_error', return_train_score=True, refit=True)
        randomized_search.fit(processed_train_set, train_set_labels)
        joblib.dump(randomized_search,'saved_objects/RandomForestRegressor_randomizedsearch.pkl')
        print_search_result(randomized_search, model_name = "RandomForestRegressor")
    
    else:
        randomized_search = joblib.load('saved_objects/RandomForestRegressor_randomizedsearch.pkl')
        print_search_result(randomized_search, model_name = "RandomForestRegressor")

# In[8]: PART 7. ANALYZE AND TEST YOUR SOLUTION
search = joblib.load('saved_objects/ElasticNet_gridsearch.pkl')
best_model = search.best_estimator_
# Pick Linear regression
#best_model = joblib.load('saved_objects/LinearRegression_model.pkl')

print('\n____________ ANALYZE AND TEST YOUR SOLUTION ____________')
print('SOLUTION: ' , best_model)
store_model(best_model, model_name="SOLUTION")   

# 7.2 Analyse the SOLUTION to get more insights about the data
# NOTE: ONLY for rand forest
if type(best_model).__name__ == "RandomForestRegressor":
    # Print features and importance score  (ONLY on rand forest)
    feature_importances = best_model.feature_importances_
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    feature_names = train_set.columns.tolist() + onehot_cols
    for name in cat_feature_names:
        feature_names.remove(name)
    print('\nFeatures and importance score: ')
    print(*sorted(zip( feature_names, feature_importances.round(decimals=4)), key = lambda row: row[1], reverse=True),sep='\n')

# %%
#full_pipeline = joblib.load(r'models/full_pipeline.pkl')

processed_test_set = full_pipeline.transform(test_set)  

# 7.3.1 Compute R2 score and root mean squared error
r2score, rmse = r2score_and_rmse(model, processed_test_set, test_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
# 7.3.2 Predict labels for some test instances
#print("\nTest data: \n", test_set.iloc[0:9])
print("\nPredictions: ", best_model.predict(processed_test_set[0:9]).round(decimals=1))
print("Labels:      ", list(test_set_labels[0:9]),'\n')
# %%
