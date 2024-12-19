import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from io import StringIO
import pydotplus
from IPython.display import Image

# Load the dataset
data = pd.read_csv('data/DogBreeds.csv')

#Define features; X and target variable; Y
X = data.drop('Type', axis = 1)
y = data['Type']

#Identify categorical data and numerical colums
#Filters colums in the DataFrame X that are of type object (strings and non-numeric data)
categorical_cols = X.select_dtypes(include=['object']).columns
#Filters out colums in DataFrame X that are of type integer och float. ".columns" extracts the column names
numerical_cols = X.select_dtypes(include = ['int64', 'float64']).columns

# Preprocessing pipelines. 
# The pupeline constructor (from scikit-learn) takes a list of steps/tuples consisting of a string identifier
# and a transformation function or estimator.
# OneHotEncoder converts categorical data into binary matrix. handle_unkown = "ignore" will ignore a new category that has not been trained. 
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown = 'ignore'))])

# Transformation function from scikit-learn for standardization.
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine preprocessing methods
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Create the model pipeline
model = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('classifier', DecisionTreeClassifier(criterion = "entropy", max_depth = 3))
 ])

# Split the data into training and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 30)

# Train model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


# Extract the trained decision tree classifier
clf = model.named_steps['classifier']

# Get feature names after preprocessing
feature_names = numerical_cols.tolist() + \
    model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_cols).tolist()

# Export the decision tree visualization
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_names, class_names=clf.classes_)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('DecisionTree.png')
Image(graph.create_png())
