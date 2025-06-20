import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import yaml
#  max_features=yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']
# ------------------------------------------------------------------
# 1. Load processed data
train_df = pd.read_csv('./data/interim/train_processed.csv')
test_df  = pd.read_csv('./data/interim/test_processed.csv')

# 2. Split into X (text) and y (label)
X_train_text = train_df['clean_comment'].astype(str)
y_train      = train_df['category'].values

X_test_text  = test_df['clean_comment'].astype(str)
y_test       = test_df['category'].values

# 3. Bag‑of‑Words
vectorizer       = CountVectorizer(max_features=50)
X_train_bow      = vectorizer.fit_transform(X_train_text)
X_test_bow       = vectorizer.transform(X_test_text)
vocab            = vectorizer.get_feature_names_out()     # column names

# 4. Convert sparse matrices to DataFrames and append label
train_bow_df = pd.DataFrame(X_train_bow.toarray(), columns=vocab)
train_bow_df['label'] = y_train

test_bow_df  = pd.DataFrame(X_test_bow.toarray(), columns=vocab)
test_bow_df['label']  = y_test

# 5. Save
output_dir = 'data/processed'
os.makedirs(output_dir, exist_ok=True)

train_bow_df.to_csv(os.path.join(output_dir, 'train_bow.csv'), index=False)
test_bow_df.to_csv(os.path.join(output_dir, 'test_bow.csv'),  index=False)
