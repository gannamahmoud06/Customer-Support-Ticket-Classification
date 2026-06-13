import os
import joblib
import pandas as pd
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_and_evaluate_model(name, model, X_train, y_train, X_test, y_test):

    print(f"Training {name}...")
    
    needs_dense = ['KNN', 'ADA']
    if any(x in name for x in needs_dense):
        X_tr = X_train.toarray() if issparse(X_train) else X_train
        X_te = X_test.toarray() if issparse(X_test) else X_test
    else:
        X_tr, X_te = X_train, X_test

    # Training and prediction
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    
    return y_pred, acc

def run_model_training_suite(all_models, X_train, y_train, X_test, y_test):

    if not os.path.exists('best_models'):
        os.makedirs('best_models')

    final_accuracies = {}     
    best_family_score = {} 
    
    for name, model in all_models.items():
        family_name = name.split('_')[0] 
        
        try:
            # Start training
            y_pred, acc = train_and_evaluate_model(name, model, X_train, y_train, X_test, y_test)
            final_accuracies[name] = acc
            
            print(f" {name} Accuracy: {acc*100:.2f}%")

            # Select best model from it family
            if family_name not in best_family_score or acc > best_family_score[family_name]:
                best_family_score[family_name] = acc
                
                # save the model
                model_path = f'best_models/best_{family_name}_model.pkl'
                joblib.dump(model, model_path)
                print(f"New Best for {family_name} saved!")
            else:
                print(f"{name} is not better than current best {family_name}.")

        except Exception as e:
            print(f"❌ {name} Failed: {e}")
            
    return final_accuracies