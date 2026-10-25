@echo off
REM One-time setup for Windows.  Double-click or run:  setup.bat
python -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required, found ' + sys.version" || exit /b 1
python -m venv .venv || exit /b 1
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install --no-cache-dir -r requirements.txt || exit /b 1
echo.
echo Setup complete. Try:  .venv\Scripts\python -m halludetect.step1_claims
echo Web app:              .venv\Scripts\streamlit run app.py
