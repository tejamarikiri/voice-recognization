from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, LeaveOneOut
import numpy as np
import joblib

def train_model(X_train, X_test, y_train, y_test):
    # Print dataset sizes
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training accuracy: {train_accuracy}")
    print(f"Test accuracy: {test_accuracy}")

    # Only calculate additional metrics if we have both classes in the test set
    if len(np.unique(y_test)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary')
        print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")
    else:
        print("Test set contains only one class. Unable to calculate precision, recall, and F1-score.")

    # Perform cross-validation if we have enough samples
    if X_train.shape[0] >= 5:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {np.mean(cv_scores)}")
    elif X_train.shape[0] > 1:
        # Use Leave-One-Out cross-validation for very small datasets
        loo = LeaveOneOut()
        cv_scores = cross_val_score(model, X_train, y_train, cv=loo)
        print(f"Leave-One-Out cross-validation scores: {cv_scores}")
        print(f"Mean LOO CV score: {np.mean(cv_scores)}")
    else:
        print("Not enough samples for cross-validation.")

    # Save the model
    joblib.dump(model, 'models/voice_detection_model.joblib')

    return model