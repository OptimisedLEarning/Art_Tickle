---
title: "Encode with Precision, Achieve Machine Learning Vision"
datePublished: Tue Apr 02 2024 01:01:33 GMT+0000 (Coordinated Universal Time)
cuid: cluhoblnw000108l749xgfkss
slug: encode-with-precision-achieve-machine-learning-vision
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1712020550031/bb9f197b-9976-4a8c-b333-d05d25f1f1ed.jpeg
tags: data-science, categorical-data

---

Encoders are <mark>used to convert categorical data into a format that can be understood by machine learning algorithms.</mark> The choice of encoder depends on the nature of the categorical data and the specific requirements of the machine learning model. Here are some common types of encoders and their use cases:

1. **One-Hot Encoding**: This method is used when the categorical variable is nominal (i.e., <mark>the categories do not have a natural order</mark>). It creates a binary column for each category. It's useful for nominal variables <mark>but can lead to a high-dimensional dataset </mark> if there are many unique categories.
    

```plaintext
**Example**: Encoding a "color" feature with categories "red", "blue", and "green" into three binary columns.
```

* **Label Encoding**: This method assigns a unique integer to each category. It's suitable for ordinal variables (<mark>where the categories have a natural order</mark>). However, it can introduce an arbitrary ordering of the categories, which might not be meaningful\*."This encoder works the <mark> same </mark> as <mark>ordinal encoding and </mark> *<mark>should only</mark>* be <mark>used when the target variable (Y) is categorical."</mark>\*.
    

```plaintext
**Example**: Encoding a "size" feature with categories "small", "medium", and "large" into the integers 0, 1, and 2.
```

3. **Ordinal Encoding**: <mark>Similar to label encoding</mark> but specifically designed for nominal variables. It assigns integers to categories based on their order, which can be useful for nominal variables that have a natural ordering.
    

```plaintext
**Example**: Encoding a "priority" feature with categories "low", "medium", and "high" into the integers 0, 1, and 2.
```

4. **Binary Encoding**: This method first converts the category *into* numerical labels <mark>using label encoding, then converts these integers </mark> into <mark>binary code</mark>. It's useful for nominal variables with many categories, as it reduces the dimensionality of the data.
    

```plaintext
**Example**: Encoding a "category" feature with 100 unique categories into binary code, reducing the dimensionality from 100 to 7 (since 2^7 = 128, which is more than enough to represent 100 categories).
```

5. **Feature Hashing**: This method applies a hash function to the features and uses the hash values as indices directly. It's handy for categorical features with <mark>lots of unique values</mark>. However, it <mark>can lead to collisions where different categories map to the same hash value</mark>, potentially <mark>losing</mark> information.
    

```plaintext
**Example**: Encoding a "user_id" feature with millions of unique IDs into a fixed-size vector using a hash function.
```

6. **Target Encoding**: This method involves <mark>replacing</mark> a categorical value with the <mark>mean of the target </mark> variable. It's helpful for categorical features with many different values, but it can cause overfitting if not used carefully.
    

```plaintext
**Example**: Encoding a "region" feature in a sales dataset with many unique regions by replacing each region with the average sales value for that region.
```

```python
import pandas as pd 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression 
from category_encoders import BinaryEncoder, OneHotEncoder, TargetEncoder
from category_encoders import HashingEncoder
# Sample data   this is not accurate data the main thing to focus 
# on what kind of data categories  have be used  

data = pd.DataFrame({ 'user_id': ['user123', 'user456', 'user789', 'user001'],
                        'gender': ['M', 'F', 'M', 'F'], 
                        'region': ['North', 'South', 'East', 'West'],
                        'age': [25, 30, 35, 40],
                        'color': ['red', 'blue', 'green', 'red', 'blue', 'green'],
                        'shape': ['circle', 'square', 'triangle', 'circle', 'square', 'triangle'] })

#Define the column transformer

preprocessor = ColumnTransformer( 
transformers=[ ('binary', BinaryEncoder(), ['gender']), 
            ('onehot', OneHotEncoder(sparse=False,drop='first'), ['region']), 
                ('label', LabelEncoder(), ['color']),
               ('ordinal', OrdinalEncoder(categories='auto'), ['shape']),# we have to manuall define categories in a list
             # ('tnf2',OrdinalEncoder(categories=[['Mild','Strong']]),['cofee']),
  # by  specifying the  we  are telinng to give bigger no. to stong and smaller to mild      
                ('target', TargetEncoder(), ['age']) ]),
                ('hashing', HashingEncoder(cols=['user_id'], n_components=8), ['user_id'])

#this line will convert the data into  data frame 
sklearn.set_config(transform_output="pandas")

# split data 
#  X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:2], df.iloc[:,-1], test_size=0.2)                           

#Define the pipeline

pipeline = Pipeline(steps=[ 
    ('preprocessor', preprocessor), 
    ('classifier', LogisticRegression()) ])

#Fit the pipeline to the data

# X_train and y_train should be your training data and labels

# pipeline.fit_transform(X_train, y_train)
# Apply the preprocessing
data_encoded = preprocessor.fit_transform(data)

# Convert the output to a DataFrame for easier viewing
data_encoded_df = pd.DataFrame(data_encoded, columns=preprocessor.get_feature_names_out())

print(data_encoded_df)
```