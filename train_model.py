import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
import pickle
import numpy as np

# Load the dataset
df = pd.read_csv('nairobi_tweets_labeled.csv')

# Preprocess data
X = df['Tweet']
y = df['Label']

# Convert text data to numerical data with adjusted feature reduction
vectorizer = CountVectorizer(max_features=100, min_df=3)  # Increased to 100
X = vectorizer.fit_transform(X)

# Feature selection - slightly more features
selector = SelectKBest(chi2, k=30)  # Increased to 30
X = selector.fit_transform(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Train the model with adjusted alpha
model = MultinomialNB(alpha=2.0)  # Reduced from 5.0 to 2.0
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5)

# Create results table
results_data = {
    'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 
               'CV Mean Score', 'CV Std Dev'],
    'Value': [f"{accuracy * 100:.2f}%", f"{f1:.4f}", f"{precision:.4f}", 
              f"{recall:.4f}", f"{np.mean(cv_scores) * 100:.2f}%", f"{np.std(cv_scores):.4f}"]
}
results_df = pd.DataFrame(results_data)

# Print results
print("\n=== Model Performance Metrics ===")
print(results_df.to_string(index=False))
print("\nConfusion Matrix:")
print(pd.DataFrame(conf_matrix, 
                  index=['Actual: No Civil Unrest', 'Actual: Civil Unrest'],
                  columns=['Pred: No Civil Unrest', 'Pred: Civil Unrest']).to_string())
print("\nCross-validation Scores:")
print(pd.Series(cv_scores, index=[f"Fold {i+1}" for i in range(5)]).to_string())
print("\nModel Configuration Notes:")
print("- Vectorizer max_features increased to 100")
print("- Minimum document frequency set to 3")
print("- Feature selection increased to top 30 features")
print("- Naive Bayes alpha reduced to 2.0")
print("- Test size kept at 35%")

# Save the model, vectorizer, and selector
with open('civil_unrest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
with open('feature_selector.pkl', 'wb') as selector_file:
    pickle.dump(selector, selector_file)