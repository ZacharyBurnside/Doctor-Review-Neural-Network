# Doctor Review Sentiment Classifier

A neural network built from scratch using only NumPy that classifies doctor reviews as positive or negative. Trained on **4.5 million real reviews** from HealthGrades and Vitals across NY, FL, and CA.

No PyTorch. No TensorFlow. Just math.

---

## Results

| Metric | Value |
|--------|-------|
| Training reviews | 80,000 |
| Test reviews | 20,000 |
| Sources | HealthGrades, Vitals |
| States | NY, FL, CA |
| Final test accuracy | **91.9%** |
| Random baseline | 50% |

---

## How it works

### Step 1 — Text becomes numbers

A neural network can't read words. Every review gets converted into a vector of 501 numbers — one count per vocabulary word, plus the review's total length.

```
"Dr. Smith was incredibly thorough and listened to my concerns"

 doctor  office  never  recommend  thorough  listened  rushed  rude  ... length
    0       0      0        0          1         1       0      0    ...    9
```

Stop words (`the`, `and`, `a`, `I`...) are stripped first — they appear in every review regardless of sentiment and just dilute the signal.

### Step 2 — Forward pass

The 501 numbers flow through two layers of neurons. Each neuron multiplies every input by a learned weight, sums them, and passes the result through an activation function.

```
Hidden layer (64 neurons, ReLU activation)
    z = x₁w₁ + x₂w₂ + ... + x₅₀₁w₅₀₁ + bias
    output = max(0, z)

Output layer (1 neuron, sigmoid activation)
    probability = 1 / (1 + e^-z)
    → number between 0 and 1
```

Above 0.5 → positive. Below 0.5 → negative.

### Step 3 — Training

The network starts with random weights and is wrong most of the time. Training fixes that:

```
for each batch of 256 reviews:
    1. predict sentiment (forward pass)
    2. measure how wrong it was (loss)
    3. figure out which weights caused the error (backprop)
    4. nudge every weight slightly in the right direction
repeat 20 times over all 80,000 reviews
```

---

## Training curve

```
Epoch      Loss   Train Acc   Test Acc
----------------------------------------
     1    0.6402       81.6%      81.3%
     2    0.4836       87.0%      86.9%
     3    0.3632       88.9%      88.7%
     4    0.3031       90.9%      90.6%
     5    0.2721       91.2%      90.9%
    10    0.2243       92.0%      91.6%
    15    0.2113       91.9%      91.5%
    20    0.2042       92.4%      91.9%
```

By epoch 2 it's already at 87%. By epoch 5 it's essentially converged. The gap between train and test accuracy stays tiny throughout — the model is generalizing, not memorizing.

---

## Sample predictions

```
✓ [NEG → NEG]  1.00   "No empathy for a sad outcome. Horrible."
✓ [NEG → NEG]  0.96   "She rushed the whole appointment. Never called me back."
✓ [NEG → NEG]  0.96   "Front office rude and highly unprofessional."
✓ [POS → POS]  0.98   "Good examination. Took time, explained everything to me."
✓ [POS → POS]  1.00   "I would highly recommend Dr. Rotatori to anyone."
✗ [NEG → POS]  0.71   "Treated a Retinal vein occlusion with laser surgery..."
```

### The one interesting miss

The network predicted positive on a negative review mentioning laser surgery. Why? Words like `surgery`, `treated`, `laser` appear heavily in grateful patient reviews — *"he performed surgery and saved my life"*. The model learned that association correctly from the data.

The problem is **bag-of-words has no word order**. It can't distinguish:

```
"treated my condition successfully"   →  treated=1, successfully=1
"treated me badly after the surgery"  →  treated=1, surgery=1, badly=1
```

The word `treated` pushes positive in both cases. This is the fundamental ceiling of the approach.

---

## Try it yourself

```python
predict("Dr. Smith was incredibly thorough and actually listened to my concerns")
# → POSITIVE  (97.0% confident)

predict("Waited two hours, spent 4 minutes with me, didn't answer any questions")
# → NEGATIVE  (87.2% confident)

predict("Rude staff, dismissive doctor, will never go back")
# → NEGATIVE  (97.4% confident)

predict("Best doctor I have ever had. Saved my life.")
# → POSITIVE  (91.4% confident)
```

---

## Word influence

Words with the strongest pull in each direction after training:

**Push positive** — `recommend`, `thorough`, `listened`, `caring`, `compassionate`,
`excellent`, `wonderful`, `attentive`, `knowledgeable`, `professional`

**Push negative** — `never`, `rude`, `horrible`, `waste`, `dismissed`,
`rushed`, `terrible`, `worst`, `ignored`, `negligent`

These aren't hand-coded rules — the network figured them out on its own by seeing 80,000 labeled examples.

---

## Data

| Source | Reviews | Rating format |
|--------|---------|---------------|
| HealthGrades | ~3.6M | 1.0 – 5.0 (float) |
| Vitals | ~900K | 1 – 5 (int) |

3-star reviews are dropped — too ambiguous to be reliable training signal. Classes are balanced to 50/50 before training so the network can't cheat by always guessing positive.

**Rating distribution (raw):**
```
1★  ██████████████  660,820
2★  █               56,537
3★  █               58,151   ← dropped
4★  ██             117,353
5★  ████████████████████████████████  3,632,611
```

---

## Setup

```bash
pip install numpy pandas
```

```bash
python doctor_sentiment.py
```

Update the file paths at the top to point to your local CSV files:

```python
HG_PATH = 'data/HEALTHGRADES_REVIEWS.csv'
VT_PATH = 'data/VITALS_REVIEWS.csv'
```

---

## Architecture

```
Input layer       501 neurons   (500 vocab word counts + review length)
     ↓
Hidden layer       64 neurons   (ReLU activation)
     ↓
Output layer        1 neuron    (sigmoid → probability 0 to 1)
```

Total trainable parameters: `501×64 + 64 + 64×1 + 1 = 32,129`

---

## Why it plateaus at ~92%

The ceiling isn't the network — it's the feature representation. Bag-of-words loses word order, context, and negation. To go higher:

- **TF-IDF** — downweights common words across all reviews, not just stop words
- **Bigrams** — treat word pairs (`not good`, `never again`) as single features
- **Word embeddings** — represent words as dense vectors that capture meaning
- **PyTorch + pretrained model** — fine-tune something like BERT on this dataset, likely pushing past 96%

---

## What's next

- [ ] TF-IDF vectorization
- [ ] Specialty-specific models (cardiology vs dermatology)
- [ ] Predict star rating 1–5 instead of binary
- [ ] Port to PyTorch
- [ ] Fine-tune a pretrained language model
