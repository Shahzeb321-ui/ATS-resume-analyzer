# Resume Fit Predictor

A Streamlit app that compares an uploaded resume with a job description using the local classifier in `resume-fit-final`.

## Run locally

1. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

2. Start the app from this folder:

   ```powershell
   streamlit run app.py
   ```

The app supports PDF, DOCX, and TXT resumes. Scanned PDFs need OCR before their text can be analyzed.
