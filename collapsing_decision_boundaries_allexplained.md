# Collapsing Decision Boundaries — Beginner's Complete Guide

Everything explained in plain English, with analogies and examples. No prior background needed.

---

## Table of Contents

- [Collapsing Decision Boundaries — Beginner's Complete Guide](#collapsing-decision-boundaries--beginners-complete-guide)
  - [Table of Contents](#table-of-contents)
  - [1. What Is This Project About?](#1-what-is-this-project-about)
  - [2. Response Time and Choices — The Raw Data](#2-response-time-and-choices--the-raw-data)
  - [3. Speed-Accuracy Trade-off (SAT)](#3-speed-accuracy-trade-off-sat)
  - [4. The Diffusion Decision Model (DDM)](#4-the-diffusion-decision-model-ddm)
  - [5. The Four Key Parameters](#5-the-four-key-parameters)
  - [6. How the Simulation Works (Euler-Maruyama)](#6-how-the-simulation-works-euler-maruyama)
  - [7. Decision Boundaries — Fixed vs Collapsing](#7-decision-boundaries--fixed-vs-collapsing)
  - [8. The Four Boundary Types This Project Tests](#8-the-four-boundary-types-this-project-tests)
  - [9. Urgency Gating — A Different Mechanism](#9-urgency-gating--a-different-mechanism)
  - [10. What Is Simulation-Based Inference (SBI)?](#10-what-is-simulation-based-inference-sbi)
  - [11. Key SBI Terms Explained](#11-key-sbi-terms-explained)
  - [12. BayesFlow — The Software](#12-bayesflow--the-software)
  - [13. The Full Project Pipeline, Step by Step](#13-the-full-project-pipeline-step-by-step)
    - [Step 1 — Define the Prior](#step-1--define-the-prior)
    - [Step 2 — Implement the Boundary Functions](#step-2--implement-the-boundary-functions)
    - [Step 3 — Build the Simulator](#step-3--build-the-simulator)
    - [Step 4 — Configure BayesFlow](#step-4--configure-bayesflow)
    - [Step 5 — Train the Networks](#step-5--train-the-networks)
    - [Step 6 — Run Simulation-Based Calibration (SBC)](#step-6--run-simulation-based-calibration-sbc)
    - [Step 7 — Fit the Empirical Data](#step-7--fit-the-empirical-data)
    - [Step 8 — Posterior Predictive Checks (PPC)](#step-8--posterior-predictive-checks-ppc)
  - [14. How We Check If the Model Is Good](#14-how-we-check-if-the-model-is-good)
  - [15. Glossary](#15-glossary)

---

## 1. What Is This Project About?

Imagine you are on a quiz show. A question flashes on the screen and a countdown timer starts ticking. As the timer approaches zero, you feel increasing pressure to just pick an answer, even if you are not fully sure. You might accept a lower level of certainty just to give any answer before time runs out.

This project tries to **mathematically describe that psychological process**. Specifically it asks:

- When people are under time pressure, do they change *how much evidence* they need before committing to a decision?
- If yes, does that threshold drop in a straight line, or in some curved shape?
- Can a computer automatically figure out the shape of that drop just by looking at how fast and accurately people responded?

The study it builds on is **Katsimpokis, Hawkins & van Maanen (2020)**. They had people watch a screen full of moving dots and quickly decide: are most dots going left or right? Some people were just *told* to be fast. Others faced a hard computerised deadline. The study found that these two types of pressure lead to mathematically different decision strategies — and this project tries to push that finding further by testing more complex mathematical shapes.

---

## 2. Response Time and Choices — The Raw Data

Every time a participant makes a decision in the experiment, two things are recorded.

**Response Time (RT)**
This is simply how long it took from the moment the stimulus appeared to the moment the person pressed a button. It is measured in seconds, for example 0.45 seconds. RT data is always *right-skewed* — meaning most responses come quickly (between 0.3 and 0.7 seconds), but occasionally someone takes 1.5 seconds or more. If you drew a histogram of RTs, it would have a tall peak on the left and a long thin tail stretching to the right.

**Choice**
Was the answer correct or incorrect? This is recorded as 1 (correct) or 0 (error).

Together, RT and choice tell us about the speed-accuracy trade-off: how fast someone went, and whether going that fast cost them accuracy.

**Non-responses** happen when a hard deadline passes before the person responds. In that case the trial records RT = missing and choice = no response. The model must reproduce non-responses at roughly the same rate as the real data.

> **Example data row:** Participant sees dots moving RIGHT. They press the "right" button after 0.38 seconds. Recorded as: RT = 0.38, choice = 1 (correct).

---

## 3. Speed-Accuracy Trade-off (SAT)

The speed-accuracy trade-off is the fundamental tension in every decision task: going faster leads to more errors. You cannot maximise both speed and accuracy at the same time.

> **Analogy:** Imagine signing your name. If someone gives you 10 seconds, your signature looks perfect. If someone gives you half a second, it is a scrawl. Every perceptual decision the brain makes works the same way — the faster you force it, the sloppier the result.

The Katsimpokis et al. experiment tested two different ways of applying speed pressure:

**Speed Instruction condition:** The experimenter simply tells participants to respond as fast as possible. There is no timer on screen. The pressure is purely psychological and verbal. The study found that under this condition, a *fixed* decision threshold fits the data best — people just commit faster overall, but they do not lower their standards as time passes within each trial.

**Response Deadline condition:** A hard cutoff is enforced by the computer. If the response does not arrive before the deadline, it is not recorded and the participant gets a penalty. Under this condition, a *linearly collapsing* threshold fits better — people appear to gradually lower their evidence requirements as the deadline approaches.

This project asks: can we do even better by testing curved (non-linear) collapsing shapes?

---

## 4. The Diffusion Decision Model (DDM)

The DDM is the main mathematical model used to describe how people make decisions. It belongs to a family called **Evidence Accumulation Models (EAMs)**, which all share the same core idea: your brain collects noisy pieces of evidence over time and commits to a decision once enough evidence has piled up.

> **Analogy — The bank account:** Imagine a bank account that starts at zero. Every millisecond, a small random amount of money is deposited or withdrawn (the noise), but overall there is a slight upward trend (the signal). As soon as the balance reaches +£100 (the upper boundary), you decide "right!" As soon as it hits −£100 (the lower boundary), you decide "left!" The time it takes to reach either limit is your Response Time. The noisier the market, or the weaker the signal, the longer it takes and the more likely you are to randomly drift the wrong way.

The mathematical description of this process is:

```
dX(t) = v · dt + σ · dW(t)
```

In plain English this equation says: the tiny change in evidence at each moment equals the signal strength multiplied by the tiny time step, plus some random noise. The `dW(t)` term is called a **Wiener process** — a mathematical description of pure random jiggling, like the path of a pollen grain drifting in water (Brownian motion).

---

## 5. The Four Key Parameters

A **parameter** is a number that controls how the model behaves. Fitting the model to data means finding which numbers best explain what the participant actually did.

**Drift Rate (v)**
How fast evidence accumulates on average. A high drift rate means the signal is strong — the task is easy, or the stimulus is clear. A low drift rate means the signal is weak — the task is hard, or the stimulus is ambiguous. Drift rate can be negative, meaning evidence on average drifts toward the wrong answer (the person is systematically confused). Typical range: roughly −3 to +3.

> **Example:** Drift rate v = 2.0 means evidence accumulates quickly toward the correct response. Most decisions will be fast and accurate. Drift rate v = 0.3 means evidence barely moves — the person is struggling, responses are slow, and errors are common.

**Boundary Separation (a)**
The distance from the neutral starting point to each decision threshold. A large boundary means the person is being very cautious — they wait until a lot of evidence has piled up before committing. This leads to slow but accurate decisions. A small boundary means the person is impulsive — they commit after very little evidence, leading to fast but error-prone decisions. This is the parameter that *collapses over time* in the models this project tests.

> **Example:** Boundary a = 2.0 means you need a lot of evidence to decide. Boundary a = 0.5 means you will commit after very little evidence — very fast, but risky.

**Non-Decision Time (t₀)**
Not all of the measured response time is spent actually deciding. Some time is spent perceiving the stimulus (encoding it into the brain) and some time is spent physically pressing the button (motor execution). The non-decision time captures this "overhead." It is subtracted from the total RT to get the time actually spent accumulating evidence.

> **Example:** If total RT = 0.55 seconds and t₀ = 0.20 seconds, then the brain spent 0.35 seconds on the actual decision process.

**Starting Point (z)**
Where does evidence accumulation begin? If z = 0.5, the process starts exactly halfway between the two boundaries — perfectly unbiased. If z > 0.5, the process starts closer to the upper (correct) boundary, meaning the person is already leaning toward one response before seeing anything. This models prior bias or response tendencies.

> **Example:** If a person always tends to say "right" when uncertain, their starting point z will be estimated above 0.5.

---

## 6. How the Simulation Works (Euler-Maruyama)

We cannot solve the DDM equation perfectly — the mathematics becomes intractable, especially with collapsing boundaries. Instead, we **approximate it in tiny time steps**. This is called the Euler-Maruyama method.

Here is what happens in each simulated trial:

**Step 1 — Set the clock to zero and evidence to the starting value.**
X starts at (z − 0.5) × 2a. If z = 0.5 and a = 1.5, X starts at 0.

**Step 2 — Each 1 millisecond, update the evidence.**
New evidence = old evidence + (drift × 0.001) + (noise × random jiggle)
The random jiggle comes from a standard normal random number scaled by the square root of the time step.

**Step 3 — Check whether a boundary has been crossed.**
If X ≥ +boundary → record RT = current time + t₀, record choice = correct.
If X ≤ −boundary → record RT = current time + t₀, record choice = error.
If the deadline passes with no crossing → record non-response.

**Step 4 — Repeat for hundreds of simulated trials.**
This produces a synthetic dataset of (RT, choice) pairs that we can compare to real data.

> **Analogy:** Like a video game that cannot run physics continuously, so it recalculates positions 60 times per second. The smaller the time step (dt), the more accurate the simulation — but smaller dt means more computation.

---

## 7. Decision Boundaries — Fixed vs Collapsing

The decision boundary B(t) is the amount of evidence required to make a decision at time t. This is what changes across the models.

**Why would the boundary collapse?**
Imagine you are searching for your keys before leaving for work. After 2 minutes of searching, you are still looking for them in the "right" place. But as time passes and you become late, you lower your standards — you start checking places you are less sure about. You are effectively reducing the threshold of certainty you require. The brain does something analogous under deadline pressure.

A collapsing boundary means: as time passes within a trial, the person requires *less* evidence before committing to a response. This naturally produces faster responses toward the end of a deadline period, at the cost of more errors.

---

## 8. The Four Boundary Types This Project Tests

**Fixed Boundary**
The threshold never changes. B(t) = a (the constant boundary value). The person needs the same amount of evidence at 0.1 seconds as at 1.5 seconds. This fits the speed instruction condition well.

Formula: `B(t) = a`

> **Analogy:** A judge who always requires exactly 10 pieces of evidence to convict, regardless of how long the trial has gone on.

---

**Linear Collapsing Boundary**
The boundary drops at a constant rate, like a straight ramp going downward. A floor prevents it from reaching zero entirely.

Formula: `B(t) = max(a − k × t, ε)`

where k is the collapse rate and ε is the minimum floor value.

> **Analogy:** A judge who starts out requiring 10 pieces of evidence, but for every extra hour the trial goes on, requires one fewer piece. After 5 hours they only need 5 pieces.

---

**Exponential Collapsing Boundary**
The boundary drops sharply and quickly at first, then slows down as it approaches zero. It collapses most aggressively in the early part of a trial.

Formula: `B(t) = a × exp(−λ × t)`

where λ (lambda) is the decay rate. A larger lambda means faster early collapse.

> **Analogy:** Radioactive decay. A substance loses half its atoms very rapidly at first, then the rate of loss slows as fewer atoms remain. The boundary behaves the same way — it drops fast at the start, then gradually approaches a low stable level.

---

**Sigmoid (S-shaped) Collapsing Boundary**
The boundary stays near its starting value for a while, then drops sharply around a specific moment in time, then levels off again. This models a "patience then panic" pattern.

Formula: `B(t) = a × (1 − 1 / (1 + exp(−k × (t − t_mid))))`

where k is the steepness of the collapse and t_mid is the time at which the sharpest drop occurs.

> **Analogy:** A student doing an exam. For the first 45 minutes they are calm and careful, requiring solid reasoning before writing anything. In the last 5 minutes they panic and start writing anything that seems vaguely related. The transition between calm and panic is the sigmoid's inflection point.

---

## 9. Urgency Gating — A Different Mechanism

Urgency gating is a fundamentally different idea from collapsing boundaries, even though the behavioural result can look similar.

In a collapsing boundary model, the *threshold itself* drops — you require less evidence as time passes.

In urgency gating, the *threshold stays fixed*, but an urgency signal is added directly to the accumulated evidence. As time passes, urgency boosts the evidence value, making it easier to cross the fixed threshold.

Formula: The decision fires when `|X(t) + u(t)| ≥ a`, where `u(t) = g × t` (linear urgency) or `u(t) = g × t²` (quadratic urgency).

> **Analogy:** Adrenaline. You are trying to lift a heavy box. The weight (threshold) does not change. But as danger increases, your body floods with adrenaline that amplifies your strength (the urgency signal). Your effective capacity increases, not the required load. The result — eventually crossing the threshold — looks the same from outside, but the mechanism is different.

**Why does this distinction matter?**
The two mechanisms make different predictions about *error response times*. Under collapsing boundaries, error RTs tend to be fast (the boundary came down and caught a random fluctuation early). Under urgency gating, late errors are also possible (urgency pushed weak evidence over the threshold late in the trial). Comparing these predictions to real data is how we can distinguish the mechanisms.

---

## 10. What Is Simulation-Based Inference (SBI)?

Here is the core technical problem: for collapsing boundary shapes, we cannot write a formula for the probability of any particular response time. The mathematics simply does not work out to a closed-form expression. This means we cannot use classical methods like Maximum Likelihood Estimation, which require computing exactly that probability.

**SBI bypasses this problem entirely.** Instead of computing probabilities, SBI generates millions of synthetic datasets at different parameter settings and trains a neural network to recognise the pattern: "when data looks like this, the parameters were probably around those values."

> **Analogy — The camera settings problem:** Imagine you are trying to figure out what camera settings (shutter speed, ISO, aperture) were used to take a photograph. You have no formula for this. But if you take one million photos at every possible combination of settings and show them all to a neural network along with the settings used, the network can eventually look at any new photo and instantly estimate the likely settings. SBI does exactly this for brain parameters. The "photos" are RT datasets, and the "settings" are the DDM parameters.

The three ingredients of SBI are:

**Prior distribution:** Your belief about what values the parameters could plausibly take before you see any data. For example: "I think drift rate v is probably somewhere between −3 and +3." This is encoded as a probability distribution.

**Simulator:** A function that takes a set of parameters and produces synthetic data. For us, this is the Euler-Maruyama DDM simulator. Given (v, a, t₀, z, λ), it outputs hundreds of (RT, choice) pairs.

**Neural network:** Trained on millions of (parameters, data) pairs, it learns to invert the relationship — given data, output the likely parameters (the posterior).

---

## 11. Key SBI Terms Explained

**Prior Distribution p(θ)**
Your belief about parameters *before* seeing data. It encodes what is plausible. A wide prior means "I am very uncertain." A narrow prior means "I am fairly confident the parameter is near this value." Getting the prior right is important — if the true parameters are outside the prior's range, the network will never learn them.

> **Example:** "I believe drift rate v is probably near 0 with most values between −3 and +3" → encoded as a Normal distribution with mean 0 and standard deviation 2.

**Posterior Distribution p(θ | data)**
Your *updated* belief after seeing the data. This is what we actually want. It is a full distribution over parameter values — not just a single best guess but an entire range of plausible values with associated probabilities.

> **Example:** "After seeing this participant's 200 trials, drift rate v is probably between 1.1 and 1.9, most likely around 1.5."

**Likelihood p(data | θ)**
How probable is this specific observed dataset if these are the true parameters? Classical fitting methods need this. SBI methods do not — they simulate instead of computing it. For collapsing boundaries, the likelihood is intractable (cannot be written as a formula), which is exactly why SBI is used.

**Amortized Inference**
The word "amortize" means to spread a cost over time. In amortized inference, you pay the training cost once (which can take hours), and then you can analyse any new participant's data instantly — the cost is amortised across all future uses. This is the main practical advantage over methods that require fresh computation for every new dataset.

**Intractable Likelihood**
"Intractable" means mathematically impossible or prohibitively expensive to compute exactly. For complex boundary shapes, the probability of any particular RT cannot be written as a simple formula. This is what forces us to use SBI rather than classical methods.

---

## 12. BayesFlow — The Software

BayesFlow is a Python library that implements SBI. It has two main neural network components that work together.

**Summary Network (SetTransformer)**
This network takes in a list of 200+ (RT, choice) pairs from one participant's session and compresses them into a short fixed-length vector — typically 64 numbers — that captures the statistical essence of that person's behaviour patterns. This is necessary because the inference network needs a fixed-size input, but different participants may have different numbers of trials.

> **Analogy:** A doctor does not read every word of a 500-page patient file. They skim for patterns: "mostly healthy, one elevated blood pressure reading, history of migraines." That short summary is what they reason from. The summary network does the same for RT data — it turns hundreds of messy trial rows into a compact, information-dense fingerprint.

**Inference Network (Normalizing Flow / InvertibleNetwork)**
This network takes the 64-number fingerprint from the summary network and produces a full posterior distribution — an entire cloud of likely parameter values, not just a point estimate. It is a special type of neural network that learns to transform a simple random distribution (like a bell curve) into the exact complex shape the posterior should have.

> **Analogy:** Imagine squeezing Play-Doh through a custom mould. The normalising flow is the mould. It takes a simple blob (a standard Gaussian) and reshapes it into the exact complex shape the posterior should have for this particular dataset.

**Training the Networks**
Both networks are trained jointly. You generate perhaps 100,000 simulation pairs (each pair is: random parameters drawn from the prior + the synthetic dataset produced by running the DDM with those parameters). The networks learn to map from dataset patterns to parameter distributions. Training might take 30 minutes to 2 hours depending on hardware. Afterwards, analysing any new participant takes less than a second.

---

## 13. The Full Project Pipeline, Step by Step

Here is the complete workflow from start to finish.

### Step 1 — Define the Prior

Decide what values each parameter could plausibly take. Write these as probability distributions. Then run a **prior predictive check**: sample some parameters from the prior, simulate some data, and ask "does this synthetic data look at all like real human RT data?" If the synthetic RTs are all 5 seconds long or if accuracy is always 100%, the prior is wrong and needs adjustment.

### Step 2 — Implement the Boundary Functions

Write Python code for each boundary shape: fixed, linear, exponential, sigmoid. For urgency gating, write the urgency signal function. These functions take time as input and return the current boundary value (or urgency value).

### Step 3 — Build the Simulator

Write the Euler-Maruyama DDM simulator. It takes a vector of parameters (v, a, t₀, z, plus any boundary-specific parameters like λ or k) and returns a set of (RT, choice) pairs. This is the scientific core of the project — it must correctly implement the physics of evidence accumulation.

### Step 4 — Configure BayesFlow

Set up the summary network (SetTransformer), the inference network (InvertibleNetwork), and wrap the prior and simulator in BayesFlow's framework. These are combined into an "amortizer" that represents the full learned mapping from data to posterior.

### Step 5 — Train the Networks

Run the training loop. BayesFlow repeatedly samples parameters from the prior, runs the simulator, and updates the network weights to improve its posterior estimates. Monitor the training loss — it should decrease steadily and then level off. If it does not converge, the networks may need to be larger, or more simulations may be needed.

### Step 6 — Run Simulation-Based Calibration (SBC)

Before touching real data, verify that the trained networks are trustworthy. SBC works as follows:

Draw 1000 sets of "true" parameters from the prior. For each, simulate a dataset and then ask the trained network: "Given this data, what do you think the parameters are?" For each case, record the rank of the true parameter among the 1000 posterior samples.

If the network is well-calibrated, the rank should be essentially random — the true value should land at rank 250 just as often as rank 750. If ranks cluster at the extremes, the posterior is systematically biased.

Plot these ranks as histograms. A well-calibrated model produces flat (uniform) histograms. Humped or U-shaped histograms indicate problems.

### Step 7 — Fit the Empirical Data

Load the actual experimental data from Katsimpokis et al. (2020). For each participant in each condition (speed instruction / response deadline), pass their trial data through the trained BayesFlow amortizer. This produces a posterior distribution over parameters — a cloud of plausible parameter values for that specific person under that specific condition.

> **Example output:** "Participant 12, deadline condition: drift rate v is most likely 1.4 (95% credible interval: 0.9 to 1.9), boundary a is most likely 0.8 (0.6 to 1.1), λ is most likely 1.7 (1.1 to 2.5)."

### Step 8 — Posterior Predictive Checks (PPC)

Now check whether the fitted model actually reproduces what participants did. Sample 200 parameter sets from the posterior, simulate 500 trials with each, and compare the resulting synthetic data distribution to the real data.

Key comparisons to make:

**RT quantile plots:** Divide responses into the 10th, 25th, 50th, 75th and 90th percentile of RT. Does the model reproduce these quantiles correctly — for correct responses and for error responses separately? The simulated quantiles should closely track the empirical ones.

**Conditional Accuracy Function (CAF):** Sort trials by RT into five bins (fastest 20%, next 20%, etc.). In each bin, compute accuracy. Does accuracy increase with RT (fast responses should be less accurate — the classic pattern)? Does the model reproduce this shape?

**Non-response rate:** What fraction of trials hit the deadline with no response? The model should match the empirical rate.

**Defective CDFs:** Plot the cumulative distribution function of RTs separately for correct and error responses. The model's simulated CDFs should overlap the empirical ones.

---

## 14. How We Check If the Model Is Good

Since we are not computing a likelihood, we cannot use standard information criteria like AIC or BIC for model comparison. Instead we rely on two main tools.

**Posterior Predictive Checks (PPC)**
These are qualitative but very informative. If the model's PPC closely matches the real data across all the plots above — RT quantiles, accuracy, non-response rate, CAF — the model is a good description of the data. If there are systematic discrepancies (for example, the model always underestimates error RTs), something is wrong with the model.

**Simulation-Based Calibration (SBC)**
This checks whether the posteriors are statistically valid — not whether they fit the data well, but whether the uncertainty they express is honest. A model can fit the data well but still have miscalibrated posteriors (for example, always being overconfident). SBC catches this.

**Qualitative model comparison**
If you train separate BayesFlow models for each boundary type (fixed, linear, exponential, sigmoid, urgency), you can visually compare their PPCs. Whichever boundary type produces simulated data that most closely matches the real data across all plots is the best-fitting model. You can also check whether the pattern replicates Katsimpokis et al.'s finding: fixed boundary best for speed instruction, collapsing boundary best for deadline condition.

---

## 15. Glossary

**EAM — Evidence Accumulation Model**
Any model where decisions happen by gradually accumulating noisy evidence until a threshold is crossed. The DDM is the most popular EAM.

**DDM — Diffusion Decision Model**
The specific EAM used in this project. Treats evidence as a random walk with a constant drift (signal) plus noise.

**RT — Response Time**
How long from stimulus appearance to the participant pressing a button. Measured in seconds.

**SAT — Speed-Accuracy Trade-off**
The fundamental tension: going faster leads to more errors. Every decision involves a trade-off between the two.

**Drift Rate (v)**
How fast evidence accumulates on average. High v = easy task, fast accurate decisions.

**Boundary Separation (a)**
How much evidence is needed to commit to a decision. Large a = cautious and slow. Small a = impulsive and fast.

**Non-Decision Time (t₀)**
Time spent on perception and motor execution — not on the actual decision. Subtracted from total RT.

**Starting Point (z)**
Where evidence accumulation begins. z = 0.5 means unbiased. z ≠ 0.5 means prior bias toward one response.

**Prior Distribution**
Your beliefs about parameter values before seeing any data. Encoded as a probability distribution.

**Posterior Distribution**
Updated beliefs about parameter values after seeing the data. What the model estimates the true parameters to be, including uncertainty.

**Likelihood**
How probable the observed data is given specific parameter values. Needed in classical methods, not in SBI.

**Intractable Likelihood**
A likelihood that cannot be written as a simple formula or computed efficiently. This is why SBI is used instead of classical methods for collapsing boundary models.

**SBI — Simulation-Based Inference**
A family of methods that replace likelihood computation with simulation. Train a network on (parameter, data) pairs; then use it to invert the mapping for any new dataset.

**Amortized Inference**
Train once on many simulations, then query the trained network instantly for any new participant. The up-front cost is amortized across all future uses.

**BayesFlow**
A Python library implementing SBI. Contains a summary network (to compress data) and an inference network (to produce posteriors).

**Summary Network**
The BayesFlow component that compresses hundreds of (RT, choice) trial rows into a short fixed-length fingerprint vector.

**Inference Network / Normalizing Flow**
The BayesFlow component that converts the compressed data fingerprint into a full posterior distribution over parameters.

**SBC — Simulation-Based Calibration**
A diagnostic test. Sample true parameters from the prior, simulate data, get the posterior, check whether the true value's rank among posterior samples is uniformly distributed. Flat rank histograms mean the posterior is well-calibrated.

**PPC — Posterior Predictive Check**
A model evaluation tool. Sample parameters from the fitted posterior, simulate new data, compare simulated data to real data. If they match, the model is a good description of behaviour.

**CAF — Conditional Accuracy Function**
Accuracy plotted as a function of RT quantile bin. Fastest responses should be least accurate; slowest responses most accurate. Used to check whether the model captures this pattern.

**Non-response**
A trial where the participant did not respond before the deadline. The model must produce non-responses at the same rate as the empirical data.

**Euler-Maruyama Method**
A numerical technique for simulating the DDM in tiny time steps (e.g., 1 millisecond each). At each step, the evidence is updated by drift plus noise. When the evidence crosses a boundary, a decision is recorded.

**Wiener Process**
A mathematical description of pure random jiggling — like the path of a pollen grain in water. Used to model the noise in the DDM's evidence accumulation.

**Fixed Boundary**
B(t) = a. The threshold stays constant over the entire trial.

**Linear Collapsing Boundary**
B(t) = max(a − k×t, ε). The threshold drops at a constant rate like a straight ramp.

**Exponential Collapsing Boundary**
B(t) = a × exp(−λ×t). The threshold drops sharply at first, then slowly approaches zero.

**Sigmoid Collapsing Boundary**
B(t) = a × (1 − sigmoid(k×(t − t_mid))). The threshold stays high for a while, then drops sharply around time t_mid.

**Urgency Gating**
A mechanism where the boundary stays fixed but an urgency signal is added to the accumulated evidence over time. The decision rule becomes: |X(t) + u(t)| ≥ a.

**Parameter**
A number that controls how a model behaves. Fitting a model means finding which parameter values best explain the observed data.

**Posterior Predictive Check**
See PPC above.

**Credible Interval**
The Bayesian equivalent of a confidence interval. A 95% credible interval means: given the data, there is a 95% probability the true parameter lies within this range.