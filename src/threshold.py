import numpy as np
from sklearn.metrics import f1_score

def find_optimal_threshold(train_df, predicted_column='past', target_column='future', num_samples=None):
    y_real = train_df[target_column].values
    y_score = np.array(train_df[predicted_column].values)

    # Number of samples in "future"
    num_future_samples = len(train_df)

    if num_samples is not None:
        # Ensure that the desired number of samples is not greater than available samples
        num_samples = min(num_samples, num_future_samples)

        # Sample from "future" to align the number of rows in y_real and y_score
        sampled_idx_future = np.random.choice(num_future_samples, size=num_samples, replace=False)
        sampled_idx_past = np.repeat(sampled_idx_future, y_score.shape[1])

        y_real = np.repeat(y_real[sampled_idx_future], y_score.shape[1])
        y_score = np.array(y_score)[sampled_idx_past]

    # Convert probabilities to binary predictions using a threshold (0.5 in this case)
    y_pred_bin = (y_score >= 0.5).astype(int)

    print("y_real shape:", y_real.shape)
    print("y_pred_bin shape:", y_pred_bin.shape)

    # Calculate micro F1-score for binary predictions
    micro_f1 = f1_score(y_real, y_pred_bin, average='micro')

    return micro_f1
