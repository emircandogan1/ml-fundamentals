import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import shutil

# get data
import tarfile
import urllib.request
from pathlib import Path
from zlib import crc32
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,cross_val_score
from scipy.stats import binom

# explore data
import matplotlib.gridspec as gridspec
import seaborn as sns
from pandas.plotting import scatter_matrix

# prepare data
from sklearn.ensemble import IsolationForest # not used (for outliers (optional))
from sklearn.preprocessing import OrdinalEncoder # not used (for similarity)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# promise models 
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, NuSVR

# fine-tune your model 
from sklearn.model_selection import GridSearchCV

# evaluate system 
from scipy import stats
import joblib

np.random.seed(seed=82)

## GET DATA ##
def data_loading():
    tarball_path = Path("data\insurance.csv")
    if not tarball_path.is_file():
        Path("data").mkdir(parents=True, exist_ok=True)
        url = "https://www.kaggle.com/datasets/mirichoi0218/insurance"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as insurance_tarball:
            insurance_tarball.extractall(path="data")
    return pd.read_csv(Path("data\insurance.csv"))

df = data_loading()

df.head()
df.info()
df["sex"].value_counts()
df.describe()

PARENT_DIR = Path("output")
DIR = "images"
path = PARENT_DIR / DIR
path.mkdir(parents=True, exist_ok=True)

def save_images(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    fig_path = path / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)

df.hist(bins=10, figsize=(12,8))
save_images("numeric_att_hist")
plt.show()

# creating test set, first shuffle than split
def shuffle_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = shuffle_split_data(df, 0.2)
print(len(train_set), len(test_set))

# hash for fetch and update new data for train and test set (need unique identifier)
def id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio*2**32

def split_data_hash(data, test_ratio, id_col):
    ids = data[id_col]
    in_test_set = ids.apply(lambda id_: id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

df_with_id = df.reset_index()
train_set, test_set = split_data_hash(df_with_id,0.2,"index")

# pretty similar shuffle_data_split function but there is biase problem (fully random instead of stratified sampling)
train_set, test_set = train_test_split(df,test_size=0.2,random_state=82)

# chance that if full random sample choosed from population where female ratio lower than %48 or lower higher than %55
sample_size = 1000 
ratio_female = 0.49
proba_too_small = binom(sample_size, ratio_female).cdf(480-1)
proba_too_large = 1 - binom(sample_size, ratio_female).cdf(555)
print(proba_too_small+proba_too_large)

# biased prevention: creating stratum for bmi's
df["bmi_cat"] = pd.cut(df["bmi"],
                      bins = [15., 30., 45.,50., np.inf],
                      labels = [1,2,3,4])

df.head()

df["bmi_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("bmi category")
plt.ylabel("Number of people")
save_images("bmi_cat_bar_plot")
plt.show()

df["bmi_cat"].value_counts()

# cross validation with stratifiedshufflesplit, we have 10 split(samples) to use train model (cross validation) best fold will give best RMSE score
splitter = StratifiedShuffleSplit(n_splits=10,test_size=0.2,train_size=0.8,random_state=82)
strat_splits = []

for train_index, test_index in splitter.split(df, df["bmi_cat"]):
    strat_train_set_n = df.iloc[train_index]
    strat_test_set_n = df.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]

# easy way to do
strat_train_set, strat_test_set = train_test_split(df,test_size=0.2,train_size=0.8,random_state=82,shuffle=True,stratify=df["bmi_cat"])

# test set created above
strat_test_set["bmi_cat"].value_counts() / len(strat_test_set)

# instead of random, use stratified (random biased, stratified much lower biased)
for set_ in(strat_train_set, strat_test_set):
    set_.drop("bmi_cat",axis=1,inplace=True)

### EXPLORE DATA / GAIN INSIGHTS ### 
# only working with train set for not data snooping in test set!!!!
strat_train_set.head()

df = strat_train_set.copy()
df.head()
df.describe()
df.info()
df["region"].value_counts()
df.isna().sum()

# categorical values plotting with countplot
sns.set_style('darkgrid')

fig = plt.figure(layout="constrained",figsize=(18,10))
grid = gridspec.GridSpec(ncols=4,nrows=2,figure=fig)

# ax1
ax1 = fig.add_subplot(grid[0, :2])
ax1.set_title("sex count")

sns.countplot(data=df,x='sex',ax=ax1,hue='sex')

# ax2
ax2 = fig.add_subplot(grid[0, 2:])
ax2.set_title("region count"),

sns.countplot(data=df,x='region',order=df['region'].value_counts().index,ax=ax2,hue='region')

#ax3 
ax3 = fig.add_subplot(grid[1, :2])
ax3.set_title('smoker count')

sns.countplot(data=df,x="smoker",order=df["smoker"].value_counts().index,ax=ax3,hue="smoker")

#ax4 
def func(pct, allvalues):
    total = sum(allvalues)
    val = int(round(pct / 100. * total))
    return f'{pct:.1f}%\n({val:,})'

ax4 = fig.add_subplot(grid[1, 2:])
ax4.set_title("Smoking Pie Chart")
smoker = df["smoker"].value_counts()
ax4.pie(smoker,labels=smoker.index,autopct=lambda pct: func(pct, smoker),shadow=True,startangle=90,textprops={"size":"larger"})
save_images('categorical_att_with_countplot')
ax4.legend()

# categorical with y attribute
fig = plt.figure(layout="constrained",figsize=(18,10))
grid = gridspec.GridSpec(ncols=4,nrows=2,figure=fig)

ax1 = fig.add_subplot(grid[0, :2])
ax1.set_title('Region-Bmi')
sns.barplot(data=df,x="bmi",y="region",hue="region",estimator="mean",order=df.groupby("region")["bmi"].mean().sort_values(ascending=False).index)

ax2 = fig.add_subplot(grid[0, 2:])
ax2.set_title('Smoker-Bmi')
sns.barplot(data=df,x="bmi",y="smoker",hue="smoker",estimator="mean")

ax3 = fig.add_subplot(grid[1, :2])
ax3.set_title("Region-Charges")
sns.barplot(data=df,x="charges",y="region",hue="region",estimator="mean",ax=ax3)

ax4 = fig.add_subplot(grid[1, 2:])
ax4.set_title("sex-charges")
sns.barplot(data=df,x="charges",y="sex",hue="sex",estimator="mean")
save_images('categorical_att_with_barplot')

# numeric histplot 
fig = plt.figure(layout="constrained",figsize=(18,10))
grid = gridspec.GridSpec(ncols=4,nrows=2,figure=fig)

ax1 = fig.add_subplot(grid[0, :2])
ax1.set_title('charges histogram')
sns.histplot(data=df,x="charges",kde=True,ax=ax1)

ax2 = fig.add_subplot(grid[0, 2:])
ax2.set_title('bmi histogram')
sns.histplot(data=df,x="bmi",kde=True,ax=ax2)

ax3 = fig.add_subplot(grid[1, :2])
ax3.set_title('age histogram')
sns.histplot(data=df,x="age",kde=True,ax=ax3)

ax4 = fig.add_subplot(grid[1, 2:])
ax4.set_title('childeren histogram')
sns.histplot(data=df,x="children",kde=True,ax=ax4)
save_images('numeric_att_histogram')

# relplot(bu sayede hem scatter,hem line bakılabiliyor)
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(20,10))
sns.boxplot(data=df,x="charges",y="region",hue="sex",width=.5)
ax.set_title("region_charges_boxplot")
save_images("region_charges_boxplot")
plt.show()

df.groupby(["region","sex"])["charges"].median()
df.groupby(["region","sex"])["charges"].min()
df.groupby(["region","sex"])["charges"].max()
df.groupby(["region","sex"])["charges"].quantile([0.25, 0.5, 0.75])

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(20,6))
sns.boxplot(data=df,x="charges",y="smoker",hue="sex",width=.5)
ax.set_title("Charges/Smoker boxplot")
plt.show()
save_images("charges_smoker_boxplot")

# correlation heatmap
corr = df[["age","bmi","children","charges"]].corr()
mask = np.triu(corr.corr())

fig, ax = plt.subplots(figsize=(20,5))
cmap = sns.diverging_palette(230,20,as_cmap=True)
sns.heatmap(data=corr,mask=mask,annot=True,fmt=".2f",cmap="coolwarm",linewidths=1)
ax.set_title("Correlation Between Variables")
save_images("corr_variables")
plt.show()

# pandas correlation 
scatter_matrix(frame=df,alpha=0.5,figsize=(20,10))
save_images('scatter matrix for correlation')

# attribute combination 
df["charges_per_child"] = df["charges"] / df["children"]
df["age_bmi"] = df["age"]*df["bmi"]

## PREPARE DATA ##
df = strat_train_set.drop("charges",axis=1)
df_target = strat_train_set["charges"].copy()

# categorical to numeric
df_cats = df[["sex","smoker","region"]]
encoder = OneHotEncoder(sparse_output=False)
df_cat_hot = encoder.fit_transform(df_cats)
df_cat_hot[:8]
encoder.categories_
encoder.feature_names_in_
encoder.get_feature_names_out()

# scaling with standard
df_num = df[["age","bmi","children"]]
scaler = StandardScaler()
df_num_scaled = scaler.fit_transform(df_num)

# scaling target (and inverse transform)
target_log = np.log(df_target)
model = LinearRegression()
model.fit(df[["bmi"]], target_log)
new_data = df[["bmi"]].iloc[:5]

predicts = model.predict(new_data)
predictions = np.exp(predicts)

# custom transformer
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(df_target)

# pipeline 
num_pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

num_prepared = num_pipe.fit_transform(df_num)
df_num_prepared = pd.DataFrame(num_prepared,columns=num_pipe.get_feature_names_out(),index=df_num.index)
num_pipe.steps

# categorical and numerical pipeline 
num_att = ["age","bmi","children"]
cat_att = ["sex","smoker","region"]

cat_pipe = make_pipeline(
    OneHotEncoder()
)

preprocessing = ColumnTransformer([
    ("numeric",num_pipe,num_att),
    ("categoic",cat_pipe,cat_att)
])

df_prepared = preprocessing.fit_transform(df)
df_prepared.shape
preprocessing.get_feature_names_out()

### PROMISING MODELS ###
# linear regression
lin_reg = make_pipeline(preprocessing,LinearRegression())
lin_rmses = -cross_val_score(lin_reg,X=df,y=df_target,scoring="neg_root_mean_squared_error",cv=10)
pd.Series(lin_rmses).describe()


lin_reg.fit(df,df_target)

predictions_cost = lin_reg.predict(df)
predictions_cost[:5].round(-2)
df_target.iloc[:5].values

lin_rmse = root_mean_squared_error(df_target,predictions_cost)

# decision tree
tree_reg = make_pipeline(preprocessing,DecisionTreeRegressor(random_state=82))
tree_reg.fit(df,df_target)
tree_predictions = tree_reg.predict(df)
tree_rmse = root_mean_squared_error(df_target,tree_predictions)

# cross val score decision tree
tree_rmses = -cross_val_score(estimator=tree_reg,X=df,y=df_target,scoring="neg_root_mean_squared_error",cv=10)
pd.Series(tree_rmses).describe()

lin_rmses = -cross_val_score(estimator=lin_reg,X=df,y=df_target,scoring="neg_root_mean_squared_error",cv=10)
pd.Series(lin_rmses).describe()

# random forest model 
forest_reg = make_pipeline(preprocessing,RandomForestRegressor(random_state=82))
forest_rmses = -cross_val_score(forest_reg,X=df,y=df_target,scoring="neg_root_mean_squared_error",cv=10)
pd.Series(forest_rmses).describe()

forest_reg.fit(df,df_target)
forest_pred = forest_reg.predict(df)
forest_pred[:5].round(-2)
df_target.iloc[:5].values
forest_rmses = root_mean_squared_error(df_target,forest_pred)

# randomforest, linear reg, decision tree, svr last (model evaluation) !!! 
# training error lower than validation error, so there is a model overfitting
svr_reg = make_pipeline(preprocessing,NuSVR(kernel="sigmoid",nu=0.9))
svr_reg.fit(df,df_target)
svr_predicts = svr_reg.predict(df)
rmse_svr = root_mean_squared_error(df_target,svr_predicts)

svr_rmses = -cross_val_score(svr_reg,X=df,y=df_target,scoring="neg_root_mean_squared_error",cv=10)
pd.Series(svr_rmses).describe()

### TUNE MODEL ### 
full_pipe = Pipeline([
    ("preprocessing",preprocessing),
    ("random_forest",RandomForestRegressor(random_state=82))
])

svr_reg.feature_importances_

# evaluate with test set
X_test = strat_test_set.drop("charges",axis=1)
y_test = strat_test_set["charges"].copy()

final_pred = forest_reg.predict(X_test)
final_rmse = mean_squared_error(y_test,final_pred,squared=False)
print(final_rmse)

# confidence interval %95 
confidence = 0.95 
squared_errors = (final_pred-y_test)**2
np.sqrt(stats.t.interval(confidence,len(squared_errors)-1,loc=squared_errors.mean(),scale=stats.sem(squared_errors)))

# model saving 
joblib.dump(forest_reg,"cost_insurance.pkl")

model_reloaded = joblib.load("cost_insurance.pkl")
new_data = df.iloc[:10]
predictoins = model_reloaded.predict(new_data)
