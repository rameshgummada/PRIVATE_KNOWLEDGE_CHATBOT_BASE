Clone git repo

Clopy .env.example to .env

Save the file

python3 -m venv venv_new

cd venv_new/

source bin/activate

python3 -m pip install -r requirements.txt

(venv_new) bash-3.2$ python load_documents.py 

No .docx or .pdf files found in data/documents. Add documents and re-run.
(venv_new) bash-3.2$ python load_documents.py 
  Processing: Tutorial_EDIT.pdf
  /user/<username>/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz: 100%|    (  Here all uploaded documents to DB)

    Loaded 115 chunks from Tutorial_EDIT.pdf
  Processing: Python Programming.pdf
    Loaded 60 chunks from Python Programming.pdf

Done. Total chunks in collection: 175
(venv_new) bash-3.2$ python load_confluence.py 
Spaces to sync: ['ENG', 'DOCS']

Make sure .env file should have correct Data




gcloud auth application-default login

Here credentials saved to your local (Json file)

streamlit run app.py    (Run this command.. here it open new browser with your vertex API chatbot)

