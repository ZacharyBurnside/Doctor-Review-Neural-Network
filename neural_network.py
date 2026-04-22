import numpy as np
import pandas as pd
from collections import Counter
import re

# ─────────────────────────────────────────────────────────────
# 1. LOAD & COMBINE
# Update these paths to point to your full files
# ─────────────────────────────────────────────────────────────
HG_PATH = '/users/zacharyburnside/desktop/neural_networks/data/HEALTHGRADES_REVIEWS.csv'
VT_PATH = '/users/zacharyburnside/desktop/neural_networks/data/VITALS_REVIEWS.csv'

print("Loading data...")

def load_reviews(path):
    chunks = []
    for chunk in pd.read_csv(path,
                             usecols=['REVIEW_RATING', 'REVIEW_TEXT'],
                             chunksize=100_000,
                             engine='python',        # handles malformed rows
                             on_bad_lines='skip'):   # skips them instead of crashing
        chunk['REVIEW_RATING'] = pd.to_numeric(chunk['REVIEW_RATING'], errors='coerce')
        chunk = chunk.dropna(subset=['REVIEW_RATING', 'REVIEW_TEXT'])
        chunk = chunk[chunk['REVIEW_TEXT'].astype(str).str.strip() != '']
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

hg = load_reviews(HG_PATH)
vt = load_reviews(VT_PATH)
df = pd.concat([hg, vt], ignore_index=True)

print(f"Total reviews loaded: {len(df):,}")
print(f"Rating distribution:\n{df['REVIEW_RATING'].value_counts().sort_index()}\n")

# ─────────────────────────────────────────────────────────────
# 2. LABEL
# positive (1) = 4-5 stars, negative (0) = 1-2 stars
# Drop 3-star reviews — too ambiguous
# ─────────────────────────────────────────────────────────────
df = df[df['REVIEW_RATING'] != 3].copy()
df['label'] = (df['REVIEW_RATING'] >= 4).astype(int)
df['REVIEW_TEXT'] = df['REVIEW_TEXT'].astype(str)

print(f"After dropping 3-stars: {len(df):,} reviews")
print(f"Positive (4-5★): {df['label'].sum():,}  |  Negative (1-2★): {(df['label']==0).sum():,}\n")

# ─────────────────────────────────────────────────────────────
# 3. BALANCE CLASSES
# With 5M+ rows we have plenty — cap each class at 50k so
# training doesn't take forever on a CPU
# ─────────────────────────────────────────────────────────────
MAX_PER_CLASS = 50_000

pos = df[df['label'] == 1].sample(min(MAX_PER_CLASS, (df['label']==1).sum()), random_state=42)
neg = df[df['label'] == 0].sample(min(MAX_PER_CLASS, (df['label']==0).sum()), random_state=42)
n = min(len(pos), len(neg))
pos = pos.iloc[:n]
neg = neg.iloc[:n]

df_balanced = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Training on: {len(df_balanced):,} reviews ({n:,} positive, {n:,} negative)\n")

# ─────────────────────────────────────────────────────────────
# 4. TEXT → NUMBERS (bag of words)
# Build vocabulary from top 500 most common words,
# then represent each review as a word-count vector
# ─────────────────────────────────────────────────────────────
STOP_WORDS = {
    'i','me','my','the','a','an','and','or','but','in','on','at','to','for',
    'of','with','is','was','he','she','they','it','his','her','this','that',
    'are','were','be','been','have','had','has','do','did','not','no','so',
    'if','as','by','from','we','you','your','our','their','its','very','also',
    'would','could','should','will','just','about','up','out','when','who',
    'more','all','there','been','than','which','what','one','can','get','got'
}

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

print("Building vocabulary...")
all_words = []
for text in df_balanced['REVIEW_TEXT']:
    words = clean(text)
    all_words.extend([w for w in words if w not in STOP_WORDS and len(w) > 2])

VOCAB_SIZE = 500
vocab = [w for w, _ in Counter(all_words).most_common(VOCAB_SIZE)]
word_to_idx = {w: i for i, w in enumerate(vocab)}
print(f"Top 20 words: {vocab[:20]}\n")

def vectorize(text):
    vec = np.zeros(VOCAB_SIZE)
    for w in clean(text):
        if w in word_to_idx:
            vec[word_to_idx[w]] += 1
    vec = np.append(vec, len(clean(text)))   # review length as extra feature
    return vec

print("Vectorizing reviews (this may take a minute for large datasets)...")
X = np.array([vectorize(t) for t in df_balanced['REVIEW_TEXT']])
y = df_balanced['label'].values.reshape(-1, 1)

# Normalize to 0-1 range
X_max = X.max(axis=0)          # save this — needed to normalize new reviews
X = X / (X_max + 1e-8)
print(f"Feature matrix: {X.shape}\n")

# ─────────────────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT  (80/20)
# ─────────────────────────────────────────────────────────────
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

# ─────────────────────────────────────────────────────────────
# 6. NEURAL NETWORK
# 501 inputs → 64 hidden (ReLU) → 1 output (sigmoid)
# ─────────────────────────────────────────────────────────────
input_size  = X.shape[1]
hidden_size = 64

np.random.seed(0)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, 1))

def relu(z):     return np.maximum(0, z)
def relu_d(z):   return (z > 0).astype(float)
def sigmoid(z):  return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def forward(X):
    Z1 = X @ W1 + b1;   A1 = relu(Z1)
    Z2 = A1 @ W2 + b2;  A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def bce_loss(A2, y):
    eps = 1e-8
    return -np.mean(y * np.log(A2 + eps) + (1 - y) * np.log(1 - A2 + eps))

def backward(X, y, Z1, A1, A2):
    m = X.shape[0]
    dZ2 = A2 - y
    dW2 = (A1.T @ dZ2) / m;   db2 = np.mean(dZ2, axis=0, keepdims=True)
    dZ1 = (dZ2 @ W2.T) * relu_d(Z1)
    dW1 = (X.T @ dZ1) / m;    db1 = np.mean(dZ1, axis=0, keepdims=True)
    return dW1, db1, dW2, db2

def accuracy(X, y):
    _, _, _, A2 = forward(X)
    return np.mean((A2 > 0.5).astype(int) == y) * 100

# ─────────────────────────────────────────────────────────────
# 7. TRAINING with mini-batches
# With 80k rows, doing a full forward pass each epoch is slow.
# Mini-batches: shuffle the data, train on 256 rows at a time.
# Same math, just faster and often converges better too.
# ─────────────────────────────────────────────────────────────
lr         = 0.1
epochs     = 20
batch_size = 256

print(f"{'Epoch':>6}  {'Loss':>8}  {'Train Acc':>10}  {'Test Acc':>9}")
print("-" * 40)

for epoch in range(1, epochs + 1):
    # Shuffle training data each epoch
    idx = np.random.permutation(len(X_train))
    X_shuf, y_shuf = X_train[idx], y_train[idx]

    epoch_loss = 0
    for start in range(0, len(X_train), batch_size):
        Xb = X_shuf[start:start + batch_size]
        yb = y_shuf[start:start + batch_size]
        Z1, A1, Z2, A2 = forward(Xb)
        epoch_loss += bce_loss(A2, yb)
        dW1, db1_g, dW2, db2_g = backward(Xb, yb, Z1, A1, A2)
        W1 -= lr * dW1;  b1 -= lr * db1_g
        W2 -= lr * dW2;  b2 -= lr * db2_g

    avg_loss   = epoch_loss / (len(X_train) // batch_size)
    train_acc  = accuracy(X_train, y_train)
    test_acc   = accuracy(X_test,  y_test)
    print(f"{epoch:>6}  {avg_loss:>8.4f}  {train_acc:>9.1f}%  {test_acc:>8.1f}%")

# ─────────────────────────────────────────────────────────────
# 8. RESULTS
# ─────────────────────────────────────────────────────────────
print(f"\nFinal test accuracy: {accuracy(X_test, y_test):.1f}%")
print(f"(Random baseline: 50%)\n")

print("Sample predictions:\n")
_, _, _, A2_test = forward(X_test)
test_texts  = df_balanced['REVIEW_TEXT'].values[split:]
test_labels = y_test.flatten()
test_probs  = A2_test.flatten()

for i in range(10):
    actual = "POS" if test_labels[i] == 1 else "NEG"
    pred   = "POS" if test_probs[i] > 0.5  else "NEG"
    conf   = test_probs[i] if test_probs[i] > 0.5 else 1 - test_probs[i]
    ok     = "✓" if actual == pred else "✗"
    print(f"  {ok} [{actual}→{pred}] {conf:.2f}  \"{test_texts[i][:90].strip()}\"")

# ─────────────────────────────────────────────────────────────
# 9. PREDICT YOUR OWN REVIEW
# ─────────────────────────────────────────────────────────────
def predict(text):
    vec = vectorize(text).reshape(1, -1)
    vec = vec / (X_max + 1e-8)
    _, _, _, prob = forward(vec)
    p = prob[0][0]
    label = "POSITIVE" if p > 0.5 else "NEGATIVE"
    conf  = p if p > 0.5 else 1 - p
    print(f"\n  \"{text}\"")
    print(f"  → {label}  (confidence: {conf:.1%})")

print("\n--- Try your own reviews ---")
predict("Dr. Smith was incredibly thorough and actually listened to my concerns")
predict("Waited two hours, he spent 4 minutes with me and didn't answer any of my questions")
predict("Rude staff, dismissive doctor, will never go back")
predict("Best doctor I have ever had. Saved my life.")