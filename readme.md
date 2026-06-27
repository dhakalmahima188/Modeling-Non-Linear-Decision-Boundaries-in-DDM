# Inference of Protein Secondary Structure Motifs
## Full Project Explanation: Domain Knowledge, Workflow, Requirements & Steps

**Course:** Simulation-Based Inference — TU Dortmund  
**Lecturer:** Simon Kucharsky (simon.kucharsky@tu-dortmund.de)

---

## Part 1: Biology Background — What Are We Actually Doing?

### 1.1 Proteins and Their Structure

A **protein** is a long chain of smaller building blocks called **amino acids**. There are exactly 20 types of amino acids (e.g., Alanine, Glycine, Leucine, …), each with slightly different chemical properties. The sequence of amino acids in a protein is called its **primary structure** — you can think of it like a string of 20 different letters.

But a protein doesn't stay as a flat string. It **folds** into a 3D shape. This shape determines what the protein *does* in the body — whether it becomes an enzyme that speeds up chemical reactions, a structural component of a cell membrane, a hormone like insulin, etc. The folded shape has multiple levels:

| Level | What it is | Example |
|---|---|---|
| Primary | The raw amino acid sequence | `FVNQHLCGSHL...` |
| **Secondary** | Local repeating folding patterns | Alpha-helix, beta-sheet, coil |
| Tertiary | The overall 3D shape of one chain | Globular shape of hemoglobin |
| Quaternary | Multiple chains together | Hemoglobin's 4 subunits |

**This project focuses on secondary structure.**

### 1.2 The Three Secondary Structure Types

When a protein chain folds, nearby amino acids form **hydrogen bonds** with each other. Depending on the pattern of bonding, three distinct local shapes emerge:

**Alpha-helix (α-helix)**
- The chain coils into a tight right-handed spiral, like a spring.
- Hydrogen bonds form between amino acid *i* and amino acid *i+4* along the chain.
- The side chains of the amino acids point outward from the spiral.
- Example: about 70% of hemoglobin's structure is alpha-helical. The channels that span cell membranes are often alpha-helices.
- Certain amino acids *love* forming helices (Alanine A, Leucine L, Glutamate E) while others (Glycine G, Proline P) tend to break them.

**Beta-sheet (β-sheet)**
- Two or more segments of the chain lie side by side and bond to each other laterally, forming a flat pleated sheet.
- Found in the core of many proteins like antibodies.

**Random coil**
- No regular repeating structure. Flexible loops connecting other elements.
- Not "random" in the chaotic sense — still deterministic — just not helix or sheet.

In this project, we simplify to **two categories: "alpha-helix" or "other"** (where "other" lumps beta-sheet and coil together).

### 1.3 Why Do Amino Acid Types Predict Secondary Structure?

Different amino acids have different shapes, sizes, and charges. These physical properties make them more or less compatible with the tight geometry required for a helix. For example:
- **Alanine (A)** has a tiny side chain — it fits easily into the compact helix geometry. Emission probability in alpha-helix state = 12%.
- **Proline (P)** has a ring structure that creates a kink — it is a "helix breaker." Emission probability in alpha-helix state = only 2%.
- **Glycine (G)** is so small it's too flexible — it also tends to be outside helices. Higher emission probability in "other" = 9%.

This is why just looking at the amino acid sequence gives you *probabilistic* information about secondary structure — not certainty, but meaningful signal.

### 1.4 The Inference Problem

Given an amino acid sequence (a string like `FVNQHLCGSHLVEALYLVCGERGFF…`), we want to ask: **for each position in this sequence, what is the probability that it is part of an alpha-helix?**

This is not a simple lookup. It requires:
1. Knowing how likely each amino acid is to appear in each state (the emission probabilities).
2. Knowing that helices tend to appear in **runs** — once you're in a helix, you tend to stay in one for several amino acids (transition probability of staying in helix = 90%).
3. Combining evidence from the **entire sequence** — the amino acid at position 50 tells you something about position 48, because they are likely in the same structural block.

This is exactly what a **Hidden Markov Model** formalizes.

---

## Part 2: The Hidden Markov Model (HMM)

### 2.1 What Is an HMM?

A Hidden Markov Model has two layers:

- **Hidden states** — what we want to know but can't directly see. Here: whether each position is "alpha-helix" or "other".
- **Observations** — what we can see. Here: the amino acid at each position.

The idea: the hidden states evolve over time according to a **Markov chain** (each state depends only on the previous state), and at each position the hidden state **emits** an observable symbol according to emission probabilities.

It's called "hidden" because the states are not directly observed — only the amino acids are.

### 2.2 The Specific HMM in This Project

**States:** {H = alpha-helix, O = other}

**Starting state:** Always begins in state O ("other"). The protein chain always starts outside a helix.

**Transition probabilities** (how the hidden state changes at each step):

```
From H → H:  90%   (helices tend to continue)
From H → O:  10%   (helices occasionally end)
From O → H:   5%   (a new helix occasionally starts)
From O → O:  95%   (non-helix regions tend to continue)
```

Written as a matrix:

|  | → H | → O |
|---|---|---|
| **From H** | 0.90 | 0.10 |
| **From O** | 0.05 | 0.95 |

**Emission probabilities** (what amino acid you see given the hidden state):

| AA | A | R | N | D | C | E | Q | G | H | I | L | K | M | F | P | S | T | W | Y | V |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Helix** | .12 | .06 | .03 | .05 | .01 | .09 | .05 | .04 | .02 | .07 | .12 | .06 | .03 | .04 | .02 | .05 | .04 | .01 | .03 | .06 |
| **Other** | .06 | .05 | .05 | .06 | .02 | .05 | .03 | .09 | .03 | .05 | .08 | .06 | .02 | .04 | .06 | .07 | .06 | .01 | .04 | .07 |

Notice: Alanine (A) is 12% likely in helix, only 6% in other → seeing Alanine is *evidence* for being in a helix. Glycine (G) is 4% in helix but 9% in other → evidence *against* being in a helix.

### 2.3 What We Can Simulate

Using this HMM as a **generative model**, we can simulate amino acid sequences of any length:

1. Start in state O.
2. At each position: emit an amino acid (sampled from the emission table for the current state).
3. Transition to the next state (sampled from the transition table for the current state).
4. Repeat for the desired sequence length.

This produces pairs of: **(amino acid sequence, hidden state sequence)**.

### 2.4 The Forward-Backward Algorithm — Computing Posterior State Probabilities

Given a sequence of amino acids that we *did not generate ourselves* (e.g., real insulin), we want to infer the probability that each position is a helix. This is called **posterior decoding**.

The naive approach — enumerate all possible state sequences — is exponentially expensive (2^N possibilities for a sequence of length N). The **Forward-Backward algorithm** computes the same result in O(N) time using dynamic programming.

**Forward pass (α):** Sweep left to right through the sequence. At each position t, compute:

```
α_t(H) = P(saw amino acids 1..t AND position t is in state H)
α_t(O) = P(saw amino acids 1..t AND position t is in state O)
```

This captures all evidence *before* position t. It uses the previous position's α values plus transition and emission probabilities.

**Backward pass (β):** Sweep right to left. At each position t, compute:

```
β_t(H) = P(will see amino acids t+1..N | position t is in state H)
β_t(O) = P(will see amino acids t+1..N | position t is in state O)
```

This captures all evidence *after* position t.

**Combining them:** The posterior probability that position t is a helix, given the **entire** sequence:

```
P(state t = H | full sequence) = α_t(H) × β_t(H) / [α_t(H)×β_t(H) + α_t(O)×β_t(O)]
```

The key insight: to know whether position 50 is a helix, you use both what came before (forward) *and* what comes after (backward). This is what makes the estimate **smooth** — it can't just look at the amino acid at position 50 in isolation.

**In practice:** The `hmmlearn` Python package does all of this for you with one function call. You don't implement Forward-Backward from scratch.

---

## Part 3: Where BayesFlow Comes In

### 3.1 The Problem with the Classical Approach

The Forward-Backward algorithm gives us exact posterior probabilities — **but only for the specific HMM we defined**. In practice, you would need to:
- Call `hmmlearn` separately for every new sequence.
- This is fast for one sequence but becomes slow at scale.
- More importantly: the goal of this project is to train a neural network to **learn** the mapping from sequence → posterior probabilities.

Why bother, if Forward-Backward is already exact and fast?

1. **It's a pedagogical goal**: this project demonstrates that a neural network can approximate what an exact algorithm computes — a core use case of simulation-based inference.
2. **Scalability**: once trained, the neural network does inference in a single forward pass — faster than even hmmlearn at large scale.
3. **Extensibility**: the network could generalize to HMMs with parameters that aren't fixed, or to more complex models where Forward-Backward doesn't apply.

### 3.2 The BayesFlow Setup

In standard SBI, BayesFlow infers model *parameters* θ from observed data x. Here, the setup is slightly different but analogous:

| Standard SBI | This project |
|---|---|
| θ = unknown model parameters | θ = hidden state probabilities per position |
| x = observed data | x = observed amino acid sequence |
| Prior p(θ) | Generative HMM (simulates sequences + true state probabilities) |
| Posterior q(θ\|x) | Neural network predicting P(helix at each position \| sequence) |

The "ground truth" for training is the output of Forward-Backward applied to simulated sequences. The network learns to replicate Forward-Backward's output — but from the amino acid sequence alone, without running the algorithm at test time.

### 3.3 Architecture

**Input:** A variable-length amino acid sequence. Each amino acid is one-hot encoded (20-dimensional vector per position) or embedded into a learned dense representation. Shape: (sequence_length, 20).

**Summary network:** Compresses the variable-length input into a fixed-length summary vector. Options:
- **LSTM** (Long Short-Term Memory): processes the sequence left-to-right, capturing sequential dependencies. Good default choice.
- **Bidirectional LSTM**: processes both left-to-right and right-to-left simultaneously — analogous to the forward and backward passes of the algorithm. Strongly recommended here.
- **Transformer/attention**: attends to all positions simultaneously.

Since we want a **per-position** prediction (not a single summary), the network should output a probability for each position rather than compressing to a single vector. This is more of a **sequence-to-sequence** architecture than a standard summary network.

**Inference network / output layer:** For each position, output a single scalar between 0 and 1 — the predicted probability of being in the alpha-helix state. Apply sigmoid activation to the final layer.

**Loss function:** Binary cross-entropy between predicted helix probabilities and the Forward-Backward ground-truth probabilities, averaged across positions and sequences.

---

## Part 4: Step-by-Step Implementation

### Step 1: Environment Setup

```bash
pip install numpy scipy matplotlib hmmlearn bayesflow torch
```

Key packages:
- `hmmlearn` — implements HMM and Forward-Backward algorithm
- `bayesflow` — neural posterior estimator framework
- `torch` — neural network backend

### Step 2: Define the HMM Parameters

```python
import numpy as np

# States: 0 = alpha-helix, 1 = other
AMINO_ACIDS = list('ARNDCEQGHILKMFPSTWYV')  # 20 standard amino acids
N_STATES = 2  # H, O

# Transition matrix [from_state, to_state]
TRANSITION = np.array([
    [0.90, 0.10],  # from H: stay H or go to O
    [0.05, 0.95],  # from O: go to H or stay O
])

# Starting probabilities (always start in state O = index 1)
START_PROB = np.array([0.0, 1.0])

# Emission matrix [state, amino_acid]
# Row 0 = alpha-helix, Row 1 = other
EMISSION = np.array([
    # A     R     N     D     C     E     Q     G     H     I     L     K     M     F     P     S     T     W     Y     V
    [0.12, 0.06, 0.03, 0.05, 0.01, 0.09, 0.05, 0.04, 0.02, 0.07, 0.12, 0.06, 0.03, 0.04, 0.02, 0.05, 0.04, 0.01, 0.03, 0.06],  # helix
    [0.06, 0.05, 0.05, 0.06, 0.02, 0.05, 0.03, 0.09, 0.03, 0.05, 0.08, 0.06, 0.02, 0.04, 0.06, 0.07, 0.06, 0.01, 0.04, 0.07],  # other
])

# Sanity checks
assert np.allclose(TRANSITION.sum(axis=1), 1.0), "Transition rows must sum to 1"
assert np.allclose(EMISSION.sum(axis=1), 1.0), "Emission rows must sum to 1"
```

### Step 3: Simulate Amino Acid Sequences

```python
def simulate_sequence(length):
    """
    Simulate one amino acid sequence and its hidden state sequence.
    Returns:
        aa_sequence: list of amino acid characters (length L)
        state_sequence: array of 0 (helix) or 1 (other), shape (L,)
    """
    states = []
    aas = []
    
    # Start in state O (index 1)
    current_state = 1
    
    for _ in range(length):
        states.append(current_state)
        
        # Emit amino acid from current state
        aa_idx = np.random.choice(20, p=EMISSION[current_state])
        aas.append(AMINO_ACIDS[aa_idx])
        
        # Transition to next state
        current_state = np.random.choice(2, p=TRANSITION[current_state])
    
    return aas, np.array(states)


def one_hot_encode(aa_sequence):
    """Convert list of amino acid chars to one-hot matrix (L, 20)."""
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    L = len(aa_sequence)
    encoded = np.zeros((L, 20), dtype=np.float32)
    for i, aa in enumerate(aa_sequence):
        encoded[i, aa_to_idx[aa]] = 1.0
    return encoded
```

**Test this first:** Simulate 10 sequences of length 50. Print them alongside their state sequences. Verify that helices appear in runs (not scattered randomly) — the transition probabilities should create stretches.

**Expected output pattern:**
```
Sequence:  F V N Q H L C G S H L V E A L Y L V C G E R G F F
States:    O O O O O O O O O O O O O H H H H H H H H H O O O
```
Notice how once the helix starts, it tends to persist.

### Step 4: Run Forward-Backward to Get Ground-Truth Posterior Probabilities

This is what the neural network will be trained to replicate.

```python
from hmmlearn import hmm

def get_posterior_probs(aa_sequence):
    """
    Given an amino acid sequence, run Forward-Backward to get
    P(helix at position t | full sequence) for all t.
    
    Returns: array of shape (L,) with probabilities in [0,1].
    """
    # Set up the HMM model in hmmlearn
    model = hmm.CategoricalHMM(n_components=2)
    model.startprob_ = START_PROB
    model.transmat_ = TRANSITION
    model.emissionprob_ = EMISSION
    
    # Encode the sequence as integer indices
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    obs = np.array([[aa_to_idx[aa]] for aa in aa_sequence])  # shape (L, 1)
    
    # Compute posterior state probabilities via Forward-Backward
    # posterior[t, s] = P(state s at position t | full sequence)
    log_prob, posterior = model.score_samples(obs)
    
    # Return probability of being in state 0 (alpha-helix) at each position
    return posterior[:, 0]  # shape (L,)
```

**Verify this works:** Run on a simulated sequence where you know the true states. The posterior probabilities should be high (~0.8–0.9) in regions you simulated as helix, and low (~0.05–0.15) elsewhere. They won't be binary 0/1 because of uncertainty at boundaries.

### Step 5: Generate Training Data

```python
def generate_dataset(n_sequences, min_length=50, max_length=200):
    """
    Generate n_sequences training pairs.
    Each pair: (one-hot encoded sequence, forward-backward posterior probabilities)
    """
    sequences = []
    posteriors = []
    
    for _ in range(n_sequences):
        L = np.random.randint(min_length, max_length + 1)
        aa_seq, _ = simulate_sequence(L)
        
        # Ground truth: Forward-Backward posterior
        post = get_posterior_probs(aa_seq)  # shape (L,)
        
        # Encode input
        encoded = one_hot_encode(aa_seq)  # shape (L, 20)
        
        sequences.append(encoded)
        posteriors.append(post)
    
    return sequences, posteriors  # lists of variable-length arrays
```

Generate:
- 50,000 sequences for training
- 5,000 for validation
- 1,000 for test

This is the entire training dataset — simulated from the HMM, labelled by Forward-Backward. **No real protein data is used for training.**

### Step 6: Build the Neural Network

Because sequences have variable length, use a **bidirectional LSTM** architecture in PyTorch:

```python
import torch
import torch.nn as nn

class HelixPredictor(nn.Module):
    """
    Sequence-to-sequence model: amino acid sequence → P(helix) at each position.
    """
    def __init__(self, input_dim=20, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        
        # Embedding: project 20-dim one-hot to a richer representation
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # Bidirectional LSTM: processes sequence in both directions
        # (mimics forward and backward passes of Forward-Backward algorithm)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,   # ← key: captures both past and future context
            dropout=dropout
        )
        
        # Output: for each position, predict P(helix) in [0, 1]
        self.output_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),  # ×2 because bidirectional
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # output in [0, 1]
        )
    
    def forward(self, x):
        """
        x: (batch, seq_len, 20) — padded one-hot encoded sequences
        returns: (batch, seq_len) — P(helix) at each position
        """
        x = self.embed(x)           # (B, L, hidden_dim)
        x, _ = self.lstm(x)         # (B, L, 2*hidden_dim)
        out = self.output_layer(x)  # (B, L, 1)
        return out.squeeze(-1)      # (B, L)
```

**Why bidirectional LSTM?** Because Forward-Backward uses information from both before *and* after each position. A unidirectional LSTM only sees past context. The bidirectional LSTM processes left-to-right and right-to-left simultaneously — the same logic as the algorithm.

### Step 7: Training

```python
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class ProteinDataset(Dataset):
    def __init__(self, sequences, posteriors, max_len=200):
        self.sequences = sequences
        self.posteriors = posteriors
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]   # (L, 20)
        post = self.posteriors[idx] # (L,)
        L = len(seq)
        
        # Pad to max_len
        seq_padded = np.zeros((self.max_len, 20), dtype=np.float32)
        post_padded = np.zeros(self.max_len, dtype=np.float32)
        mask = np.zeros(self.max_len, dtype=bool)
        
        seq_padded[:L] = seq
        post_padded[:L] = post
        mask[:L] = True
        
        return (
            torch.tensor(seq_padded),
            torch.tensor(post_padded),
            torch.tensor(mask)
        )

def train(model, train_loader, val_loader, n_epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        
        for seq, post_true, mask in train_loader:
            optimizer.zero_grad()
            
            post_pred = model(seq)  # (B, L)
            
            # Only compute loss on non-padded positions
            loss = F.binary_cross_entropy(
                post_pred[mask], post_true[mask]
            )
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = sum(
                F.binary_cross_entropy(model(seq)[mask], post[mask]).item()
                for seq, post, mask in val_loader
            ) / len(val_loader)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")
```

**Training tips:**
- Use batch size 32–64.
- Variable-length sequences require padding and masking (implemented above).
- ~30–50 epochs is usually enough to converge.
- Watch for the validation loss plateauing — that's when to stop.

### Step 8: Evaluation on Simulated Data

Before touching real proteins, verify that the model works on held-out simulated sequences:

**Metric 1 — Mean Absolute Error (MAE):** Average absolute difference between predicted and Forward-Backward posterior probabilities. Should be < 0.05.

**Metric 2 — Calibration:** For all positions where the model predicts P(helix) ≈ 0.7, about 70% should actually be helix in the true state sequence. Plot a calibration curve (predicted probability vs. actual fraction).

**Metric 3 — Boundary detection:** The model should be uncertain (predict ~0.5) at helix start and end points, and confident (predict > 0.8 or < 0.2) in the middle of runs. Visualise a few sequences:

```
Position:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
True state: O  O  O  H  H  H  H  H  H  H  O  O  O  O  O
FB output:  .04 .05 .08 .82 .91 .94 .93 .91 .85 .72 .15 .06 .04 .03 .04
NN output:  .05 .06 .09 .79 .89 .93 .91 .90 .83 .69 .17 .07 .05 .04 .05
```

### Step 9: Apply to Real Protein — Human Insulin (PDB: 1A7F)

Human insulin has two chains:
- **A-chain:** 21 amino acids
- **B-chain:** 30 amino acids

The annotated secondary structure from the RCSB Protein Data Bank (1A7F) shows the known helix regions. This is your **ground truth** for comparison.

**Get the sequence:**

```python
# Human insulin A-chain (21 AA)
insulin_A = "GIVEQCCTSICSLYQLENYCN"

# Human insulin B-chain (30 AA)
insulin_B = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"
```

**Run the comparison:**

```python
def compare_to_ground_truth(sequence, known_helix_positions, model):
    """
    sequence: amino acid string
    known_helix_positions: list of (start, end) tuples (0-indexed, inclusive)
    """
    # Neural network prediction
    encoded = one_hot_encode(list(sequence))
    x = torch.tensor(encoded).unsqueeze(0)  # (1, L, 20)
    with torch.no_grad():
        nn_pred = model(x).squeeze(0).numpy()  # (L,)
    
    # Forward-Backward prediction (exact)
    fb_pred = get_posterior_probs(list(sequence))  # (L,)
    
    # Ground truth binary labels
    L = len(sequence)
    true_labels = np.zeros(L)
    for start, end in known_helix_positions:
        true_labels[start:end+1] = 1.0
    
    # Plot
    positions = np.arange(1, L + 1)
    plt.figure(figsize=(14, 4))
    plt.bar(positions, true_labels, alpha=0.3, color='gray', label='Known helix (annotated)')
    plt.plot(positions, fb_pred, 'b-o', markersize=4, label='Forward-Backward (exact)')
    plt.plot(positions, nn_pred, 'r--s', markersize=4, label='Neural network (BayesFlow)')
    plt.axhline(0.5, color='k', linestyle=':', alpha=0.5)
    plt.xlabel('Sequence position')
    plt.ylabel('P(alpha-helix)')
    plt.title(f'Helix probability — {sequence[:10]}...')
    plt.legend()
    plt.tight_layout()
    plt.show()
```

**Known helix regions in insulin B-chain (approximate, from 1A7F):**
- Residues 9–19: central alpha-helix (positions 8–18 in 0-indexed)

**What you expect to see:**
- Forward-Backward and neural network predictions should closely match.
- Both should peak (high P(helix)) in the region known to be helical.
- The network won't be perfect — it learned from a *simplified* HMM, and real proteins are more complex — but it should capture the main helical regions.

---

## Part 5: What to Report

### Results Section

1. **Simulated data evaluation:** Plot of true state sequence vs. Forward-Backward vs. neural network prediction for 3–5 representative simulated sequences (one short helix, one long helix, one largely "other").

2. **Calibration curve:** Does the network's predicted probability of 0.7 actually correspond to 70% helix? (It should, if well-trained.)

3. **Loss curves:** Training and validation binary cross-entropy over epochs.

4. **Quantitative comparison on test set:**
   - MAE between neural network and Forward-Backward predictions.
   - Accuracy at each position when thresholding at 0.5.

5. **Human insulin comparison:** Side-by-side plot of forward-backward, neural network, and known structure.

### Discussion Section

1. **Why does the network work?** The bidirectional LSTM implicitly learns the same logic as Forward-Backward — using past context (forward LSTM) and future context (backward LSTM) at every position.

2. **Where does it fail?** The model was trained on a simplified 2-state HMM with fixed emission probabilities derived from aggregate statistics. Real proteins:
   - Have position-specific biases (not every helix position is equally likely to contain Alanine).
   - Fold based on 3D interactions between distant positions, not just local sequence.
   - Are influenced by the cellular environment, co-translational folding, etc.

3. **What's the advantage over simply running Forward-Backward?** At test time, once trained, the neural network is faster. More importantly, it demonstrates that neural networks can capture the computations of exact inference algorithms — a key insight in modern machine learning.

4. **Could the network generalize beyond this HMM?** Potentially yes — if you trained on diverse HMMs with different parameters, the network might learn more general features of sequence-to-structure mapping.

---

## Part 6: Summary Table

| What | Details |
|---|---|
| **Biology** | Proteins fold; secondary structure = local shape (helix, sheet, coil) |
| **Model** | 2-state HMM (helix / other) with fixed transition + emission probabilities |
| **Simulator** | Markov chain: sample states → sample amino acids from emission table |
| **Ground truth labels** | Forward-Backward algorithm via `hmmlearn` → per-position P(helix) |
| **Training data** | 50k simulated (sequence, posterior) pairs |
| **Network** | Bidirectional LSTM → sigmoid output → P(helix) per position |
| **Loss** | Binary cross-entropy between NN prediction and FB ground truth |
| **Validation** | MAE, calibration curve, visual comparison on simulated sequences |
| **Real-world test** | Human insulin (1A7F): compare predictions to annotated helix regions |

---

## Part 7: References

1. Protein structure overview: https://old-ib.bioninja.com.au/higher-level/topic-7-nucleic-acids/73-translation/protein-structure.html

2. HMM tutorial: https://scholar.harvard.edu/files/adegirmenci/files/hmm_adegirmenci_2014.pdf

3. Emission probabilities derived from: https://www.kaggle.com/datasets/alfrandom/protein-secondary-structure

4. Forward-Backward algorithm: https://web.stanford.edu/~jurafsky/slp3/A.pdf (Speech & Language Processing, Appendix A)

5. hmmlearn Python package: https://pypi.org/project/hmmlearn/

6. Human insulin structure (1A7F): https://www.rcsb.org/3d-sequence/1A7F

7. Radev, S. T., et al. (2023). BayesFlow: Amortized Bayesian workflows with neural networks. *JOSS, 8*(89), 5702.

---

## Part 8: Timeline

| Week | Task |
|---|---|
| 1 | Biology reading; implement HMM; simulate sequences; visualise state runs |
| 2 | Implement Forward-Backward via hmmlearn; verify posterior probabilities make sense |
| 3 | Build and train bidirectional LSTM; monitor training loss |
| 4 | Evaluate on test set (MAE, calibration); compare to Forward-Backward |
| 5 | Apply to human insulin; write report |