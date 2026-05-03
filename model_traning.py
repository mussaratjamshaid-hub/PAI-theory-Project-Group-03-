import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

# load dataset
data = pd.read_csv("dataset.csv")

# agar columns v1,v2 ho to uncomment karo:
# data.columns = ['label', 'text']

# clean data
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# convert labels
data = data[['Category','Message']]

data.columns = ['label','text']

data['label'] = data['label'].map({'spam':1, 'ham':0})

# split
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# models
lr = LogisticRegression()
svm = SVC()

# train
lr.fit(X_train_vec, y_train)
svm.fit(X_train_vec, y_train)

# evaluate
lr_pred = lr.predict(X_test_vec)
svm_pred = svm.predict(X_test_vec)

print("LR Accuracy:", accuracy_score(y_test, lr_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

# save
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))
pickle.dump(lr, open("model_lr.pkl","wb"))
pickle.dump(svm, open("model_svm.pkl","wb"))

print("Training Done ✅")
