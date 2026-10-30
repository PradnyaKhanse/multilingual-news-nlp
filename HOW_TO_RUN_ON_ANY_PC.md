# Instructions to Run on Any Friend's Laptop (Offline / Viva Ready)

## 📌 Option 1: If You Share the Entire Folder (Including `.venv`)
If your friend's laptop has a 64-bit Windows OS:
1. Copy the entire `PROJ NLP` folder onto a pendrive / drive.
2. Paste it anywhere on her laptop (e.g. `D:\PROJ NLP` or `C:\Users\<Name>\Downloads\PROJ NLP`).
3. Open the folder and **double-click** `run_app.bat`.
   - The app will automatically launch in her browser at `http://localhost:8501`.
   - No Python installation or command prompt typing is required!

---

## 📌 Option 2: If Running via Command Prompt (Manual Steps)

If she wants to run it from her own CMD:
1. Open **Command Prompt (cmd)**.
2. Navigate to where she pasted the project folder:
   ```cmd
   cd /d "PATH_TO_FOLDER"
   ```
   *(For example: `cd /d "C:\Users\Friend\Downloads\PROJ NLP\PROJ NLP"`)*
3. Run:
   ```cmd
   .venv\Scripts\streamlit.exe run app.py
   ```

---

## 📌 Option 3: Fresh Setup (If she creates a new environment)
If she does not want to use the included `.venv`:
1. Ensure Python 3.10 or 3.11 is installed on her laptop (check "Add Python to PATH" during installation).
2. Open CMD inside the project folder:
   ```cmd
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   .venv\Scripts\streamlit.exe run app.py
   ```

---

## ⚠️ Important Note About First-Time Model Loading
- All three transformer models:
  - **Classification:** `models/classifier/best_model` (included locally inside the project)
  - **Summarization:** `ai4bharat/IndicBART`
  - **Translation:** `facebook/nllb-200-distilled-600M`
- On her laptop, the very first time she clicks "Run Full Pipeline", if the models are not in her cache, it will download them from Hugging Face (requires internet for 2-3 mins).
- **Tip for Viva:** Run the app once on her laptop before the viva while connected to Wi-Fi/hotspot so all weights are cached locally and it runs instantly offline during the viva!
