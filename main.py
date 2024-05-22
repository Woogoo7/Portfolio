import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

# Load the data
df = pd.read_csv('data/homework.csv')

# Define columns to drop
columns_to_drop = [
    'id', 'url', 'region', 'region_url', 'price', 'manufacturer', 'image_url',
    'description', 'posting_date', 'lat', 'long'
]

# Step 1: Filter Data
class FunctionalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X)

def filter_data(df):
    return df.drop(columns_to_drop, axis=1)

# Step 2: Handle Outliers
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        boundaries = self.calculate_outliers(X['year'])
        self.lower_bound, self.upper_bound = boundaries
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[X_copy['year'] < self.lower_bound, 'year'] = round(self.lower_bound)
        X_copy.loc[X_copy['year'] > self.upper_bound, 'year'] = round(self.upper_bound)
        return X_copy

    @staticmethod
    def calculate_outliers(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
        return boundaries

# Step 3: Feature Engineering
def short_model(x):
    if not pd.isna(x):
        return x.lower().split(' ')[0]
    else:
        return x

# Step 4: Handle Missing Values and Scaling
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 5: Create Final Pipeline
final_pipeline = Pipeline(steps=[
    ('filter', FunctionalTransformer(filter_data)),
    ('outlier_remover', OutlierRemover()),
    ('feature_engineering', FunctionalTransformer(lambda x: x.assign(short_model=df['model'].apply(short_model),
                                                                     age_category=df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))))),
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())  # Initial classifier, will be updated later
])

# Step 6: Train and Evaluate Models
X = df.drop(['price_category'], axis=1)
y = df['price_category']

models = [
    LogisticRegression(solver='liblinear'),
    RandomForestClassifier(),
    SVC()
]

best_model = None
best_score = 0

for model in models:
    final_pipeline.set_params(classifier=model)
    score = cross_val_score(final_pipeline, X, y, cv=4, scoring='accuracy')
    mean_score = score.mean()

    print(f'Model: {type(model).__name__}, Accuracy Mean: {mean_score:.4f}, Accuracy Std: {score.std():.4f}')

    if mean_score > best_score:
        best_score = mean_score
        best_model = model

# Step 7: Save the Best Model
final_pipeline.set_params(classifier=best_model)
final_pipeline.fit(X, y)

# Save the pipeline to a pickle file
with open('final_pipeline.pkl', 'wb') as file:
    pickle.dump(final_pipeline, file)
