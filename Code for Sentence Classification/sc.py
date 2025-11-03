
%pip install xgboost scikit-learn pandas numpy

%pip install --upgrade --force-reinstall numpy pandas scikit-learn xgboost

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report, accuracy_score

from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder

#nedd to provide appropriate path to the dataset
data = pd.read_csv("TC-Sentences1K.csv")

print(data.head())

X = data['AWP-Sentence']
y = data['Type-of-Sentence']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

vectorizer = TfidfVectorizer(
    max_features=3000,        # limit vocabulary size
    ngram_range=(1,2),        # include unigrams + bigrams
    stop_words='english'      # remove stop words
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(set(y)),
    eval_metric='mlogloss',
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train_tfidf, y_train_enc)

y_pred = model.predict(X_test_tfidf)

y_pred_labels = le.inverse_transform(y_pred)

print("Accuracy:", accuracy_score(y_test_enc, y_pred))

print("\nClassification Report:\n", classification_report(y_test_enc, y_pred, target_names=le.classes_))
