# Poker Equity Predictor (PyTorch)

A neural network that predicts **post-flop poker hand equity** (expected share of the pot against a random opponent hand) given a hand and board state, trained on simulated data using PyTorch.

This was my first full ML project, built in my 2nd year of a BSc Mathematics degree at **Queen Mary University of London**. It demonstrates an end-to-end ML pipeline: data generation, feature engineering, model training, and evaluation.

The model focuses on post-flop situations (flop, turn, and river). Pre-flop only invovles 1,326 distinct starting hands, so a simple look-up table is more appropriate there, whereas post-flop equity depends on the specific board cards and is a more interesting modelling problem.

---

## 🔢 Key Results

- **Overall test MAE:** 0.02048  
- **Flop MAE:** 0.01550  
- **Turn MAE:** 0.01968  
- **River MAE:** 0.02639  
- **Baseline (always guessing 0.5) MAE:** 0.20789  

So the model reduces error by roughly **90%** compared to a naive 0.5 guess.

---

## 🧠 What the project does

- Uses the [`treys`](https://pypi.org/project/treys/) poker library to simulate Texas Hold’em hands
- Generates labels via:
  - **Monte Carlo simulation** on the *flop*
  - **Brute-force enumeration** on the *turn* and *river*
- Encodes each state as a **119-dimensional feature vector**:
  - 104 binary card features (hand + board)
  - Street indicator (flop / turn / river)
  - Hand strength class from the evaluator
  - Pair / suited indicators and a rank-gap feature
- Trains a feed-forward neural network in PyTorch with:
  - Batch Normalisation
  - SiLU activations
  - Early stopping based on validation MAE

---

## 📁 Files in this repo

- `poker_model.py` – Neural network architecture  
- `generate_poker_data.py` – Simulation-based dataset generation  
- `train_poker_model.py` – Feature engineering + training loop  
- `evaluate_poker_model.py` – Evaluation on a held-out test set  
- `validation_data.csv` – 10k validation hands with accurate equity labels  
- `test_data.csv` – 10k test hands with accurate equity labels  
- `poker_model_state.pth` – Trained model weights selected by best validation MAE  

All files are in the main project folder, so the scripts work with their default file paths.

---

## 📂 Data

The repository already includes:

- `validation_data.csv` – 10k hands with accurate equity labels (used for model selection)  
- `test_data.csv` – 10k hands with accurate equity labels (used for final evaluation)  

The full **10 million–hand training dataset** I used for the best results is too large to store in the repo, but you can download it here:

- 10M training set (`training_data.csv`):  
  https://drive.google.com/file/d/1Rso2CDTPqsFWjNP6-jPVoczs02MCc7pU/view?usp=sharing

After downloading, place `training_data.csv` in the main project folder (next to the Python scripts) so that `train_poker_model.py` can find it.

---

## ⚙️ Getting started

### 1. Install dependencies

I recommend installing **PyTorch** by following the official instructions for your OS / CUDA setup:

https://pytorch.org/get-started/locally/

Then install the remaining Python packages, for example:

```bash
pip install pandas treys numpy
```

---

## 🚀 Quick evaluation (recommended)

This is the easiest way to see what the model does. The pretrained model and test dataset are included in the repo.

1. Clone the repository:

```bash
git clone https://github.com/TylerColeStudent/poker-equity-predictor.git
cd poker-equity-predictor
```

2. Install dependencies (see above).

3. Run evaluation:

```bash
python evaluate_poker_model.py
```

This will:

* load `poker_model_state.pth`
* load `test_data.csv`
* compute the overall test MAE and separate MAEs for flop / turn / river
* print the results to the console

On my machine, this takes only a few seconds.

---

## 🏋️ Full pipeline: data generation + training (optional)

If you want to reproduce the full process yourself (or experiment with different settings), you can generate new datasets and retrain the model.

### A. Generate datasets

```bash
python generate_poker_data.py
```

This will create three CSV files in the repo directory:

* `training_data.csv` – simulated hands with cheap outcome labels (used for training)
* `validation_data.csv` – hands with accurate equity labels (used for model selection)
* `test_data.csv` – hands with accurate equity labels (used for final evaluation)

> ⚠️ **Runtime note**
> With the default settings, this step is computationally expensive.
> On my machine (Ryzen 5 5600X, RTX 3060 Ti, 32GB RAM) it took roughly **7 hours** to generate:
>
> * 10 million training hands
> * 10k validation hands
> * 10k test hands

If you want a quicker run, you can decrease `TRAIN_HANDS`, `VAL_HANDS` and `TEST_HANDS` in `generate_poker_data.py` (e.g. divide each by 10). However, using fewer training hands will noticeably reduce the model's accuracy.

### B. Train the model

Once `training_data.csv` and `validation_data.csv` exist, run:

```bash
python train_poker_model.py
```

This will:

* vectorise the card strings into 119-dimensional feature vectors
* train the neural network using Adam + `BCEWithLogitsLoss`
* use early stopping based on validation MAE
* save the best model weights to `poker_model_state.pth`

> ⏱ On the same hardware, full training with 10M training hands took about **2 hours** using the GPU (RTX 3060 Ti).

You can then re-run:

```bash
python evaluate_poker_model.py
```

to evaluate your newly trained model on `test_data.csv`.

---

## 💻 Hardware & runtimes

For reference, the runtimes above are from this setup:

* **CPU:** AMD Ryzen 5 5600X

* **GPU:** NVIDIA RTX 3060 Ti

* **RAM:** 32GB DDR4

* Data generation (default 10M training hands): ~**7 hours**

* Training with early stopping: ~**2 hours**

* Evaluation on 10k test hands: **a few seconds**

These heavy jobs are intended as **offline batch steps**. Once the model is trained, inference is fast.

---

I’ll continue to refine the documentation and may add smaller example datasets for quicker experimentation, but this README already contains everything needed to run evaluation or reproduce the full pipeline.

