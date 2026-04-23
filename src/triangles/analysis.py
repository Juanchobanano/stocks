import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score

from patterns import tech as tradingpatterns_tech
from data import generate_data_head_shoulder


def main():
    df = generate_data_head_shoulder(10)

    df = detect_and_rename(df, 'window', 3)
    df = detect_and_rename(df, 'filter', 3, threshold=0.01, time_delay=1)
    df = detect_and_rename(df, 'kf', 3)
    df = detect_and_rename(df, 'wavelet', 3)

    algorithms = [
        'head_shoulder_pattern_window',
        'head_shoulder_pattern_filter',
        'head_shoulder_pattern_kf',
        'head_shoulder_pattern_wavelet',
    ]

    true_labels = (
        ['Head and Shoulder'] * 10
        + ['Inverse Head and Shoulder'] * 10
        + ['No Pattern'] * (len(df) - 20)
    )

    print_algorithm_accuracies(algorithms, true_labels, df)

    patterns = ['Head and Shoulder', 'Inverse Head and Shoulder']
    ground_truth_counts = Counter(true_labels)
    predicted_counts = {alg: Counter(df[alg].fillna('No Pattern')) for alg in algorithms}

    df_counts = pd.DataFrame(index=patterns)
    df_counts['Ground Truth'] = [ground_truth_counts[p] for p in patterns]
    for alg in algorithms:
        df_counts[alg] = [predicted_counts[alg][p] for p in patterns]

    plot_heatmap(df_counts)
    plt.savefig('heatmap.png')


def detect_and_rename(df, method, window, threshold=None, time_delay=None):
    """Run the requested detection method and rename the output column to include the method name."""
    if method == 'filter':
        df = tradingpatterns_tech.detect_head_shoulder_filter(df, window, threshold, time_delay)
    elif method == 'kf':
        df = tradingpatterns_tech.detect_head_shoulder_kf(df, window)
    elif method == 'wavelet':
        df = tradingpatterns_tech.detect_head_shoulder_wavelet(df, window)
    else:
        df = tradingpatterns_tech.detect_head_shoulder(df, window)

    df.rename(columns={'head_shoulder_pattern': f'head_shoulder_pattern_{method}'}, inplace=True)
    return df


def print_algorithm_accuracies(algorithms, true_labels, df):
    """Print accuracy_score for each algorithm column against the ground-truth labels."""
    for alg in algorithms:
        predicted_labels = df[alg].fillna('No Pattern').tolist()
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f'Accuracy for {alg}: {accuracy * 100:.2f}%')


def plot_heatmap(df_counts):
    """Render a heatmap comparing ground truth vs. predicted pattern counts."""
    sns.heatmap(df_counts, annot=True, fmt="d", cmap="YlGnBu", yticklabels=['HS', 'I-HS'])
    plt.title("Pattern Counts: Ground Truth vs. Predicted")
    plt.tight_layout()


if __name__ == "__main__":
    main()
