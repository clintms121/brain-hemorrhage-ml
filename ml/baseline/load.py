from data.cleaning.process import OUTCOME_CODES, run_pipeline, split_features_target
from sklearn.model_selection import train_test_split

clean, processed = run_pipeline(verbose=False)
X, y = split_features_target(processed) # X: 5000 x 33 features, y: outcome 0-3

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

