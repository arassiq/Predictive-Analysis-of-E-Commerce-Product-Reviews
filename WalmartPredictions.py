import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt

#graphing
import matplotlib.pyplot as plt

#pre processing 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

#classification imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


#language learning
from sklearn.feature_extraction.text import CountVectorizer #NLP Specific
from rake_nltk import Rake
from sklearn.naive_bayes import MultinomialNB #A new supervised learning method 
import nltk 
from nltk.corpus import stopwords #things conjoining ideas
from nltk.tokenize import word_tokenize #creating words out of sentance (seperation of word from a sentance)

df = pd.read_csv('/Users/mob/Desktop/Folders/Junior Year Semester 2/Application of Analytics/Final Project Walmart/Womens Clothing E-Commerce Reviews.csv')

#print(df.head())

#determine which columns we need to impute/delete

'''
for column_name, series in df.items():

    if(series.isna().sum() != 0):
        print(f"Percent of data missing in {column_name}: %{((series.isna().sum()/series.shape[0])*100)}")
'''
        
'''

1) Pre process
    -convert nan in 'title' to missing
    -delete rows without 'review text'
    -impute for division name, class name, department name
2) build classification to predict reccomended IND, regression for rating
3) heat maps to map out variable correlation
4) LLM

'''

#Pre Processing / PipeLines

df = df.dropna(subset= ['Review Text'])

df = df.dropna(subset= ['Division Name'])


'''
Imputing title with Rake keyword extraction (see test file for test application)

Function applys rake to review text, retreives and scores best keywords, then returns the highest scored phrase

for loop iterates through each row where 'Title' == nan and applies the highest scored phrase to the title 
'''

rake = Rake()

def applyRake(reviewText):

    rake.extract_keywords_from_text(reviewText)
    phrases = rake.get_ranked_phrases()
    return phrases[0]

#iterrows = iterator for rows within pd, allows for more complex iteration

for index, row in df[df['Title'].isna()].iterrows():

    df.at[index, 'Title'] = applyRake(row['Review Text'])

#print(df.isna().sum())


numericalFeatures = ['Age', 'Rating']

categoricalFeatures = [ 'Department Name', 'Class Name']

textualFeatures = ['Title', 'Review Text']

categoricalTransformer = Pipeline(steps= [

    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))

])

numericalTransformerScaled = Pipeline(steps= [

    ('imputer', SimpleImputer(strategy= 'mean')),
    ('scaler', StandardScaler())
    
])

textual_transformer = Pipeline(steps=[
    ('vect', CountVectorizer(stop_words='english'))
])

'''Classification and Correlation'''

preProcessor = ColumnTransformer(
    transformers = [
        ('num', numericalTransformerScaled, numericalFeatures),
        ('cat', categoricalTransformer, categoricalFeatures)
    ]
)

preProcessorText = ColumnTransformer(
    transformers=[
        ('num', numericalTransformerScaled, numericalFeatures),
        ('cat', categoricalTransformer, categoricalFeatures),
        ('text', textual_transformer, 'Review Text')
    ]
)




X = df[['Age', 'Rating', 'Department Name', 'Class Name']]
y = df['Recommended IND']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

randomForestModel = Pipeline(steps = [
    ('preprocessor', preProcessor),
    ('classifier', RandomForestClassifier(random_state = 42))
])

randomForestModelText = Pipeline(steps = [
    ('preprocessor', preProcessorText),
    ('classifier', RandomForestClassifier(random_state = 42)),
])

randomForestModel.fit(X_train, y_train)
y_predClassification = randomForestModel.predict(X_test)

print(f"\nAccuracy of Random Forest classification model: %{round(accuracy_score(y_test, y_predClassification) * 100, 3)}\n" ) 

#Random Forest Plus Text Processing


'''Correlation'''


rfModelExtracted = randomForestModel.named_steps['classifier']

featureImportances = rfModelExtracted.feature_importances_

#print(featureImportances)

catPipeline = randomForestModel.named_steps['preprocessor'].named_transformers_['cat']

oneHotEncoder = catPipeline.named_steps['onehot']

oneHotFeatureNames = oneHotEncoder.get_feature_names_out(input_features=categoricalFeatures)
#print(oneHotFeatureNames)


allFeatureNames = numericalFeatures + oneHotFeatureNames.tolist()

featureImportancesDf = pd.DataFrame(
    featureImportances, columns=["Importance"], index=allFeatureNames
)

featureImportancesDf.sort_values(by="Importance", ascending=True).plot(
    kind="barh", figsize=(9, 7), legend=False
)

print("\n(correlation graph outputed)\n")

plt.title("Feature Importances in the RandomForest Model")
plt.xlabel("Relative Importance")
plt.ylabel("Features")
plt.subplots_adjust(left=0.3)
plt.show()


X = df[['Age', 'Rating', 'Department Name', 'Class Name', 'Review Text']] 
y = df['Positive Feedback Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

randomForestModelText.fit(X_train, y_train)
y_predClassification = randomForestModelText.predict(X_test)

print(f"\nAccuracy of Random Forest classification model (inclouding language processing): %{round(accuracy_score(y_test, y_predClassification) * 100, 3)}\n" ) 



'''Regression'''

X = df[['Age', 'Rating', 'Department Name', 'Class Name']]
y = df['Positive Feedback Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

print("\n Regression Models and their fitting (lower is better): \n")
models = [
    Pipeline(steps=[('preprocessor', preProcessor), ('regressor', LinearRegression())]),
    Pipeline(steps=[('preprocessor', preProcessor), ('regressor', RandomForestRegressor(random_state=42))]),
    Pipeline(steps=[('preprocessor', preProcessor), ('regressor', KNeighborsRegressor())])
]
for pipeline in models:
    pipeline.fit(X_train, y_train)  # X_train should contain all relevant columns, both numerical and categorical
    y_preds = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_preds)
    rmse = sqrt(mse)
    print(f"{pipeline.named_steps['regressor']} - MSE: {mse}, RMSE: {rmse}")


RegressionModel = Pipeline(steps = [
    ('preprocessor', preProcessor),
    ('regressor', LinearRegression())
])

RegressionModel.fit(X_train, y_train)

y_preds = RegressionModel.predict(X_test)

mse = mean_squared_error(y_test, y_preds)
rmse = sqrt(mse)
 
#linear regression = best
#print(f"MSE: {mse}\n")
#print(f"RMSE: {rmse}\n")


'''Visualizing residuals'''

residuals = y_test - y_preds

print("\n(Linear Regression Residual Plot Outputted - lowest RMSE)\n")

plt.figure(figsize=(10, 6))
plt.scatter(y_preds, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

'''
linear regression: residuals increase as predicted values increase. This shows that the model has a harder time predicting higher values 

randomForestRegressor: higher RMSE and MSE, larger ske
'''

''' Coefficient importance:'''

regressor = RegressionModel.named_steps['regressor']

coefficients = regressor.coef_

featureImportanceDf = pd.DataFrame({
    'Feature': allFeatureNames,
    'Coefficient': coefficients
})

#use abs() to find which variables have the largest impact
featureImportanceDf = featureImportanceDf.reindex(featureImportanceDf.Coefficient.abs().sort_values(ascending=False).index)

#print(featureImportanceDf)

def prepreocess_text(text):
    tokens = word_tokenize(text)
    #removing stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

df['Review Text'] = df['Review Text'].apply(prepreocess_text)



linearRegressionText = Pipeline(steps=[
    ('preprocessor', preProcessorText),
    ('classifier', LinearRegression())
])

X = df[['Age', 'Rating', 'Department Name', 'Class Name', 'Review Text']] 
y = df['Positive Feedback Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

""" 
linearRegressionText.fit(X_train, y_train)

y_preds = linearRegressionText.predict(X_test)

mse = mean_squared_error(y_test, y_preds)
rmse = sqrt(mse)

print(f" With text MSE: {mse}\nWith text RMSE: {rmse}")#text vectorization does not work on the regression model, instead doubling MSE and RMSE
 """

models = [
    Pipeline(steps=[('preprocessor', preProcessorText), ('regressor', LinearRegression())]),
    Pipeline(steps=[('preprocessor', preProcessorText), ('regressor', RandomForestRegressor(random_state=42))]),
    Pipeline(steps=[('preprocessor', preProcessorText), ('regressor', KNeighborsRegressor())])
]

print("\nRegression Models with Count Vectorizer:\n")

i = 0 

for pipeline in models:
    pipeline.fit(X_train, y_train)  # X_train should contain all relevant columns, both numerical and categorical
    y_preds = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_preds)
    rmse = sqrt(mse)
    print(f"{pipeline.named_steps['regressor']} - MSE: {mse}, RMSE: {rmse}")

    residuals = y_test - y_preds


    if(i == 1):

        print("(RandomForest Regressor Residual Plot Outputted)")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_preds, residuals)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

    i+= 1

#random Forest Regressor residuals decreased, not significantly, plot looks similar
    
    