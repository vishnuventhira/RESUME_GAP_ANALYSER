import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

topics = [
    # Formatting (10)
    ("formatting", "Single column layout vs multi-column for ATS compatibility"),
    ("formatting", "Font choices and sizes that ATS systems can parse correctly"),
    ("formatting", "Headers and section headings ATS systems recognise"),
    ("formatting", "Tables, text boxes, and graphics that break ATS parsing"),
    ("formatting", "File format recommendations - PDF vs DOCX for ATS"),
    ("formatting", "Page margins, spacing, and length best practices"),
    ("formatting", "Contact information placement and formatting"),
    ("formatting", "Bullet points vs paragraphs in work experience"),
    ("formatting", "How ATS systems parse dates and employment history"),
    ("formatting", "Headers and footers that ATS systems cannot read"),
    
    # Keywords (10)
    ("keywords", "How ATS keyword matching works - exact vs semantic matching"),
    ("keywords", "Where to place keywords for maximum ATS impact"),
    ("keywords", "Hard skills vs soft skills in ATS keyword matching"),
    ("keywords", "Technical skills section optimisation for ATS"),
    ("keywords", "Industry-specific keywords for data and technology roles"),
    ("keywords", "Industry-specific keywords for finance and accounting roles"),
    ("keywords", "Industry-specific keywords for marketing and communications roles"),
    ("keywords", "Industry-specific keywords for project management roles"),
    ("keywords", "Certifications and qualifications - full names vs abbreviations"),
    ("keywords", "Action verbs that ATS systems and recruiters respond to"),
    
    # Tailoring (8)
    ("tailoring", "How to analyse a job description for ATS keywords"),
    ("tailoring", "Mirroring job description language in your CV"),
    ("tailoring", "Required vs nice-to-have skills - how to prioritise"),
    ("tailoring", "Tailoring work experience bullet points to match JD requirements"),
    ("tailoring", "How experience level affects ATS scoring"),
    ("tailoring", "Career changers - how to frame transferable skills for ATS"),
    ("tailoring", "Cover letter keywords and how they complement CV keywords"),
    ("tailoring", "How to handle OR conditions in job requirements"),
    
    # Common mistakes (8)
    ("mistakes", "Top 10 reasons CVs fail ATS screening"),
    ("mistakes", "Keyword stuffing - why it backfires in modern ATS systems"),
    ("mistakes", "Using images or graphics to represent skills"),
    ("mistakes", "Inconsistent date formats that confuse ATS parsing"),
    ("mistakes", "Unusual section headings that ATS cannot categorise"),
    ("mistakes", "Missing contact information fields ATS systems expect"),
    ("mistakes", "Sending wrong file format for ATS submission"),
    ("mistakes", "How employment gaps affect ATS scoring"),
    
    # ATS Systems (7)
    ("ats_systems", "How Workday ATS parses and scores CVs"),
    ("ats_systems", "How Greenhouse ATS evaluates candidate applications"),
    ("ats_systems", "How Lever ATS processes CV submissions"),
    ("ats_systems", "How LinkedIn applies ATS-style filtering to applications"),
    ("ats_systems", "Differences between enterprise ATS and small company hiring tools"),
    ("ats_systems", "How ATS systems handle international CVs and non-English content"),
    ("ats_systems", "How modern AI-powered ATS differs from traditional keyword matching"),
    
    # Cover letters (7)
    ("cover_letter", "Cover letter structure that complements ATS keyword optimisation"),
    ("cover_letter", "Opening paragraph techniques that capture recruiter attention"),
    ("cover_letter", "How to connect achievements to job requirements in cover letters"),
    ("cover_letter", "Cover letter length and format for ATS submission"),
    ("cover_letter", "Career change cover letters - addressing gaps honestly"),
    ("cover_letter", "Cover letters for senior roles - what changes"),
    ("cover_letter", "Common cover letter mistakes that undermine strong CVs"),
]

def generate_document(category: str, topic: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": f"""Write a concise, factual ATS (Applicant Tracking System) best practices document about: {topic}

Requirements:
- 150-200 words
- Practical and actionable advice
- Specific examples where helpful
- No fluff or generic advice
- Written as reference content for a CV analysis tool

Return only the document text, no titles or headings."""
            }
        ]
    )
    
    return {
        "title": topic,
        "category": category,
        "content": response.choices[0].message.content.strip()
    }

def generate_all():
    # load existing if any
    output_path = Path("data/ats_knowledge.json")
    if output_path.exists():
        with open(output_path) as f:
            documents = json.load(f)
        existing_titles = {d['title'] for d in documents}
        print(f"Resuming — {len(documents)} already generated")
    else:
        documents = []
        existing_titles = set()
    
    total = len(topics)
    
    for i, (category, topic) in enumerate(topics, 1):
        if topic in existing_titles:
            print(f"[{i}/{total}] Skipping: {topic[:60]}")
            continue
            
        print(f"[{i}/{total}] Generating: {topic[:60]}...")
        try:
            doc = generate_document(category, topic)
            documents.append(doc)
            existing_titles.add(topic)
            print(f"  Done ({len(doc['content'])} chars)")
            
            with open(output_path, 'w') as f:
                json.dump(documents, f, indent=2)
                
        except Exception as e:
            if '429' in str(e):
                if 'tokens per day' in str(e):
                    print(f"  Daily limit hit — stopping. Resume tomorrow.")
                    break
                else:
                    print("  TPM limit — waiting 60s...")
                    time.sleep(60)
                    try:
                        doc = generate_document(category, topic)
                        documents.append(doc)
                        existing_titles.add(topic)
                        with open(output_path, 'w') as f:
                            json.dump(documents, f, indent=2)
                    except Exception as e2:
                        print(f"  Failed again: {e2}")
                        break
            else:
                print(f"  ERROR: {e}")
        
        time.sleep(3)
    
    print(f"\nDone. Total: {len(documents)} documents")

if __name__ == "__main__":
    generate_all()