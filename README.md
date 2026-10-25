# Hallucination Detection in AI-Generated Text

A hybrid, explainable detector. It splits an AI answer into claims and checks each one with three signals: Wikipedia evidence with NLI, self-consistency, and model confidence. It then combines them into a risk score with a reason for every verdict.

📖 **For the full step-by-step explanation, see [GUIDE.md](GUIDE.md).**

---

## Run it in VS Code

### 1. Requirements
- **Python 3.10 or newer**. Check with `python --version` (Windows) or `python3 --version` (Mac).
  If you need it, get it from https://www.python.org/downloads/. On Windows, tick **"Add Python to PATH"** during install.
- **VS Code** with the **Python** extension. VS Code will suggest it when you open the folder.
- About **4 GB of free disk space** (≈1.5 GB of libraries + ≈1.4 GB of AI models downloaded on first run).
- An internet connection (for model download and Wikipedia).

### 2. Open the project
Unzip, then in VS Code: **File → Open Folder… → `hallucination-detector`**

### 3. One-time setup
Open the terminal (**Terminal → New Terminal**) and run:

**Windows**
```
setup.bat
```

**macOS / Linux**
```
bash setup.sh
```

This creates a private Python environment in `.venv` and installs everything (5–10 minutes).
If VS Code then asks *"We noticed a new environment… select it?"*, click **Yes**.
Otherwise: `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the one with `.venv`.

### 4. Run
Open the **Run and Debug** panel (`Ctrl+Shift+D`), pick a configuration from the dropdown, and press ▶:

| Configuration | What it does |
|---|---|
| **▶ Run web app (Step 8)** | Starts the app. Open http://localhost:8501 in your browser |
| Step 1 … Step 6 | Runs that step's demo so you can see what it does |
| Full pipeline (Steps 1–6) | The Einstein example, end to end |
| Step 7: Train & evaluate | Downloads HaluEval, trains the combiner, prints metrics (~10 min) |

Or from the terminal, with the environment activated:

```bash
python -m halludetect.step1_claims     # any step
streamlit run app.py                   # web app
python train_and_evaluate.py --n-items 400
```

The **first run of Steps 2–8 downloads the models** (~1.4 GB), so it is slow once. After that it's fast.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found (Windows) | Reinstall Python with "Add to PATH" ticked, or use `py` instead of `python` |
| `TypeError: unsupported operand type(s) for \|` | Your Python is older than 3.10. Install a newer one |
| `ModuleNotFoundError` | VS Code is using the wrong interpreter. Select the `.venv` one (step 3) |
| PowerShell blocks `.venv\Scripts\activate` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt |
| `429 Too Many Requests` from Wikipedia | Wikipedia's rate limit. The detector retries and skips the page; wait a minute if it keeps happening |
| `Could not load models/combiner.pkl` | Different scikit-learn version. Run `python train_and_evaluate.py` to retrain |
| Very slow | Normal on CPU. Untick "Self-consistency" in the app sidebar (the slowest signal) |

## Project structure
```
halludetect/        the detector, one file per step (each has a runnable demo)
app.py              Streamlit web app
train_and_evaluate.py
models/combiner.pkl trained weights (from the 400-item HaluEval run)
results/metrics.json
GUIDE.md            step-by-step explanation, results, viva questions
```
