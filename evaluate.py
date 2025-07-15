import csv
import math
import random

def load_data(path):
    cat_maps = {
        'Sex': {'M': 1.0, 'F': 0.0},
        'ChestPainType': {'TA': 0.0, 'ATA': 1.0, 'NAP': 2.0, 'ASY': 3.0},
        'RestingECG': {'Normal': 0.0, 'ST': 1.0, 'LVH': 2.0},
        'ExerciseAngina': {'N': 0.0, 'Y': 1.0},
        'ST_Slope': {'Up': 0.0, 'Flat': 1.0, 'Down': 2.0}
    }
    features = []
    labels = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat = [
                float(row['Age']),
                float(row['RestingBP']),
                float(row['Cholesterol']),
                float(row['FastingBS']),
                float(row['MaxHR']),
                float(row['Oldpeak']),
                cat_maps['Sex'][row['Sex']],
                cat_maps['ChestPainType'][row['ChestPainType']],
                cat_maps['RestingECG'][row['RestingECG']],
                cat_maps['ExerciseAngina'][row['ExerciseAngina']],
                cat_maps['ST_Slope'][row['ST_Slope']],
            ]
            label = int(row['HeartDisease'])
            features.append(feat)
            labels.append(label)
    return features, labels

def train_test_split(features, labels, test_ratio=0.2, seed=42):
    data = list(zip(features, labels))
    random.seed(seed)
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    train = data[:split]
    test = data[split:]
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    return list(X_train), list(y_train), list(X_test), list(y_test)

def standardize(train, test):
    n_features = len(train[0])
    means = [0.0]*n_features
    stds = [0.0]*n_features
    for j in range(n_features):
        col = [row[j] for row in train]
        means[j] = sum(col) / len(col)
        var = sum((x - means[j])**2 for x in col) / len(col)
        stds[j] = math.sqrt(var) if var > 0 else 1.0
    def transform(dataset):
        return [[(row[j] - means[j]) / stds[j] for j in range(n_features)] for row in dataset]
    return transform(train), transform(test)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def train_log_reg(X, y, lr=0.1, epochs=200):
    n = len(X[0])
    weights = [0.0]*n
    bias = 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = sum(w*x for w, x in zip(weights, xi)) + bias
            yhat = sigmoid(z)
            error = yhat - yi
            for j in range(n):
                weights[j] -= lr * error * xi[j]
            bias -= lr * error
    return weights, bias

def predict(weights, bias, X):
    preds = []
    for xi in X:
        z = sum(w*x for w, x in zip(weights, xi)) + bias
        yhat = sigmoid(z)
        preds.append(1 if yhat >= 0.5 else 0)
    return preds

def evaluate(y_true, y_pred):
    tp = tn = fp = fn = 0
    for y, p in zip(y_true, y_pred):
        if y == 1 and p == 1:
            tp += 1
        elif y == 0 and p == 0:
            tn += 1
        elif y == 0 and p == 1:
            fp += 1
        else:
            fn += 1
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def main():
    X, y = load_data('notebook/data/heart.csv')
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    X_train, X_test = standardize(X_train, X_test)
    weights, bias = train_log_reg(X_train, y_train)
    preds = predict(weights, bias, X_test)
    metrics = evaluate(y_test, preds)
    for k,v in metrics.items():
        print(f'{k}: {v}')

if __name__ == '__main__':
    main()
