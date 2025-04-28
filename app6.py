import streamlit as st
import requests
import pdfplumber

API_BASE = "https://resumehunter.onrender.com"

# ---------- API Helpers ----------
def ats_check(resume_text, job_description):
    try:
        payload = {"resume_text": resume_text, "job_description": job_description}
        res = requests.post(f"{API_BASE}/ats_check/", json=payload)
        res.raise_for_status()
        return res.json().get("output", "No ATS check output.")
    except Exception as e:
        st.error(f"Error fetching ATS check: {e}")
        return "Error occurred during ATS check."

def clear_pinecone():
    try:
        res = requests.post(f"{API_BASE}/clear_pinecone/")
        res.raise_for_status()
        return res.json().get("message", "Cleared successfully.")
    except Exception as e:
        st.error(f"Error clearing Pinecone: {e}")
        return "Error occurred during clearing."

# ---------- PDF Helper ----------
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
    return "\n".join(pages)

# ---------- UI Setup ----------
st.set_page_config(page_title="ATS Resume Checker", page_icon="🤖")
st.title("📄 Resume Hunter")

# Sidebar
st.sidebar.title("⚙️ Settings")
if st.sidebar.button("Clear Pinecone DB"):
    with st.spinner("Clearing Pinecone database..."):
        message = clear_pinecone()
    st.sidebar.success(message)

# Main Page
with st.form("ats_form"):
    job_title = st.text_input("Job Title:", placeholder="e.g., Data Scientist")
    jd_text = st.text_area("Paste Job Description (JD) here:", height=100)
    uploaded_files = st.file_uploader("Upload Resume PDFs", accept_multiple_files=True, type=["pdf"])
    submitted = st.form_submit_button("Check ATS Compatibility")

    if submitted and jd_text and uploaded_files:
        results = []
        with st.spinner("Processing Resumes..."):
            for file in uploaded_files:
                resume_text = extract_text_from_pdf(file)
                ats_result = ats_check(resume_text, jd_text)
                
                # Try to extract ATS Score
                score = 0
                try:
                    score_line = [line for line in ats_result.splitlines() if "ATS Match Score" in line][0]
                    score = int(''.join(filter(str.isdigit, score_line)))
                except:
                    pass

                results.append({
                    "filename": file.name,
                    "score": score,
                    "output": ats_result
                })
        
        if results:
            # Sort by ATS score descending
            results.sort(key=lambda x: x["score"], reverse=True)

            # Display Best Resume First
            best_resume = results[0]
            st.header("🏆 Best Matching Resume")
            st.success(f"The best matching resume is **{best_resume['filename']}** with an ATS Match Score of **{best_resume['score']}%**.")
            st.write("**Reason:** This resume has the highest alignment with the job description based on skills, experiences, and relevant keywords.")

            st.markdown("---")

            # Display All Results
            st.header("📄 Detailed ATS Results")
            for res in results:
                st.subheader(f"Resume: {res['filename']}")
                st.markdown(f"**ATS Match Score:** {res['score']}%")
                st.markdown(res["output"])
                st.markdown("---")
        else:
            st.warning("No valid resumes processed.")
