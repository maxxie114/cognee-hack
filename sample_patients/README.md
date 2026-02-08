# Sample patient documents for testing

Use these three files to test **document-scoped chat** and **replace-on-ingest** in the Streamlit app.

## Files

| File | Patient | Distinct details you can ask about |
|------|---------|-------------------------------------|
| `patient_maria_santos.txt` | Maria Santos (PT-1001) | Diabetes, metformin, lisinopril, penicillin allergy |
| `patient_james_chen.txt` | James Chen (PT-1002) | Asthma, fluticasone/salmeterol, GERD, sulfa allergy |
| `patient_sarah_williams.txt` | Sarah Williams (PT-1003) | Hypothyroidism, levothyroxine, vitamin D, latex allergy |

## How to test

1. **Start** Qdrant, the API (`uv run api`), and Streamlit (`uv run streamlit run streamlit_app.py`).
2. In the **sidebar**, upload all three files from `sample_patients/` and click **Ingest into RAG**.
3. In the main area, open **"Chat with document"** and select **patient_maria_santos.txt**.
4. Ask: *"What medications does this patient take?"* or *"Does this patient have any drug allergies?"*  
   You should get only **Maria’s** info (metformin, lisinopril; penicillin allergy).
5. Switch the selector to **patient_james_chen.txt** and ask the same questions.  
   You should get only **James’s** info (asthma inhalers, omeprazole; sulfa allergy).
6. Select **patient_sarah_williams.txt** and ask again.  
   You should get only **Sarah’s** info (levothyroxine, vitamin D; latex allergy).

If answers mix patients, document scoping is not working. If you ingest again with only one file and the dropdown still shows all three, replace-on-ingest is not working.
