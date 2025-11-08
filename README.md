# Poker Equity Predictor (PyTorch)

A neural network that predicts **post-flop poker hand equity** (win probability) given a hand and board state, trained on simulated data using PyTorch.

This was my first full ML project, built in my 2nd year of a BSc Mathematics degree at **Queen Mary University of London**. It demonstrates an end-to-end ML pipeline: data generation, feature engineering, model training, and evaluation.

---

## 🔢 Key Results

* **Overall test MAE:** 0.02048
* **Flop MAE:** 0.01550
* **Turn MAE:** 0.01968
* **River MAE:** 0.02639
* **Baseline (always guessing 0.5) MAE:** 0.20789

So the model reduces error by roughly **90%** compared to a naive 0.5 guess.

---

## 🧠 What the project does

* Uses the [`treys`](https://pypi.org/project/treys/) poker library to simulate Texas Hold’em hands
* Generates labels via:

  * **Monte Carlo simulation** on the *flop*
  * **Brute-force enumeration** on the *turn* and *river*
* Encodes each state as a **119-dimensional feature vector**:

  * 104 binary card features (hand + board)
  * Street indicator (flop / turn / river)
  * Hand strength class from the evaluator
  * Pair / suited indicators and a rank-gap feature
* Trains a feed-forward neural network in PyTorch with:

  * Batch Normalisation
  * SiLU activations
  * Early stopping based on validation MAE

---

## 📁 Files in this repo

* `poker_model.py` – Neural network architecture
* `generate_poker_data.py` – Simulation-based dataset generation
* `train_poker_model.py` – Feature engineering + training loop
* `evaluate_poker_model.py` – Evaluation on a held-out test set

---

## ⚙️ Basic usage

### 1. Install dependencies

I recommend installing **PyTorch** by following the official instructions for your OS / CUDA setup:

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Then install the remaining Python packages, for example:

```bash
pip install pandas treys numpy
```

### 2. Run evaluation (once you have a model + test data)

If `poker_model_state.pth` and `test_data.csv` are present in the same directory as the scripts, you can run:

```bash
python evaluate_poker_model.py
```

to compute the MAE on the test set and see the per-street breakdown.

---

I’ll be adding downloadable datasets and the trained model, plus more detailed run instructions (including runtimes), so the full pipeline is easy to reproduce.

---

