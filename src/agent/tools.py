import json
import os
import profile
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ── LLM Instances ─────────────────────────────────────────────────────────────
llm_precise = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_json = llm_precise.bind(response_format={"type": "json_object"})
llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ── Schema ────────────────────────────────────────────────────────────────────
class JobRequirements(BaseModel):
    job_title: str
    seniority_level: str          # junior | mid | senior | lead
    domain: str                   # fintech | healthtech | ecommerce | etc
    required_skills: List[str]
    nice_to_have_skills: List[str]
    experience_years: int
    keywords: List[str]
    or_conditions: List[str]
    leadership_required: bool


# ── Tool Input Schema ─────────────────────────────────────────────────────────
class ExtractJobRequirementsInput(BaseModel):
    jd_text: str

class Education(BaseModel):
    degree: str          # "MS"
    field: str           # "Data Science"
    institution: str     # "UT Dallas"
    year: str            # "2016"


class Experience(BaseModel):
    title: str
    company: str
    duration: str        # "Jan 2023 - Present"
    highlights: List[str]


class Project(BaseModel):
    name: str
    description: str
    technologies: List[str]

class CandidateProfile(BaseModel):
    name: str
    current_title: str
    years_experience: int
    seniority_level: str                    # junior | mid | senior | lead
    has_leadership_experience: bool
    summary: str
    technical_skills: List[str]
    tools_technologies: List[str]
    education: List[Education]
    experience: List[Experience]
    certifications: List[str]
    projects: List[Project]
    notable_achievements: List[str]

class ParseCVInput(BaseModel):
    cv_text: str


class ResumeScore(BaseModel):
    match_score: int
    matched_keywords: List[str]
    missing_keywords: List[str]
    summary: str

class ScoreResumeInput(BaseModel):
    profile: dict           # output of parse_cv (CandidateProfile)
    requirements: dict      # output of extract_job_requirements (JobRequirements)


# ── System Instructions ───────────────────────────────────────────────────────
job_requirement_instructions = """You are a senior technical recruiter.
Extract structured requirements from a job description.

RULES:
- For OR conditions like "Power BI or Tableau": add BOTH to required_skills
  AND add the full phrase to or_conditions
- Set experience_years to 0 if not explicitly mentioned
- seniority_level must be one of: junior, mid, senior, lead
- leadership_required is true if JD mentions leading, managing, or mentoring people
- DO NOT include: company names, locations, generic traits like "team player"
- Only extract: technical skills, tools, certifications, domain knowledge
- Return ONLY valid JSON, no markdown, no extra text"""

parse_cv_instructions = """You are an expert resume parser.
Extract structured information from a CV/resume text.

expert resume parser that works across ALL professions
(technology, engineering, healthcare, finance, science, trades, and more).

Extract structured information from a CV/resume text.

RULES:
- name: candidate's full name, or "Unknown" if not found
- current_title: current or most recent job title
- years_experience: total years of professional experience as an integer (0 if not stated)
- seniority_level must be one of: junior, mid, senior, lead
- has_leadership_experience is true if any role mentions leading, managing, mentoring, or owning a team

SKILLS — extract comprehensively across EVERY skills subsection, not just the first line:

- technical_skills: domain knowledge, methods, techniques, and competencies the candidate
  APPLIES using expertise. These are things one learns and performs.
  Examples by field:
    • Data/Software: machine learning, statistical modeling, regression, NLP, system design
    • Mechanical Eng: thermodynamics, finite element analysis, GD&T, CAD modeling, stress analysis
    • Electrical Eng: circuit design, PCB layout, signal processing, embedded systems, power systems
    • Healthcare: patient assessment, wound care, diagnostics, clinical procedures
    • Finance: financial modeling, valuation, risk analysis, forecasting
  Include methods listed under ANY heading (e.g. "Machine Learning", "Statistics",
  "Core Competencies", "Areas of Expertise").

- tools_technologies: named software, platforms, programming languages, instruments,
  equipment, frameworks, and systems the candidate OPERATES.
  Examples by field:
    • Data/Software: Python, SQL, AWS, Tableau, TensorFlow, Docker, Snowflake
    • Mechanical Eng: SolidWorks, ANSYS, AutoCAD, MATLAB, CATIA
    • Electrical Eng: SPICE, Altium, Verilog, oscilloscopes, MATLAB/Simulink
    • Healthcare: Epic EHR, Cerner, specific medical devices
    • Finance: Excel, Bloomberg Terminal, SAP, QuickBooks
  Include programming languages here as named tools.

GENERAL RULES:
- Extract ALL items from EVERY relevant section — do not stop at the first line or category
- highlights: the bullet points UNDER each role, not the role title itself
- summary: a 2-3 sentence professional summary (use the CV's summary if present, else synthesize)
- Do NOT fabricate any information — only extract what is explicitly in the CV
- If a field is genuinely absent, return empty string "" or empty list []
- When a skill could fit either category, prefer technical_skills for methods/knowledge
  and tools_technologies for anything with a product/brand/language name
  """

score_rubric = """You are an ATS (Applicant Tracking System) simulator.
ATS software scans resumes for keywords from the job description, with some
tolerance for variants and synonyms. Simulate this realistically.

A required skill is MATCHED if it appears in the candidate profile as:
- the literal keyword, OR
- a clear synonym or variant ("ML" ≈ "Machine Learning", "deep learning" ≈ "CNN"), OR
- a specific instance of a general category
  ("XGBoost" matches "algorithms"; "Survival Analysis" matches "advanced analytical methodologies"), OR
- an obvious umbrella term the candidate's title/summary establishes
  (a "Data Scientist" satisfies "data science" and "data analysis")

A required skill is MISSING only if NO keyword, variant, synonym, instance, or
clear contextual evidence exists anywhere in the profile.

Compute match_score (0-100) as a weighted sum, each component scored 0-100:

1. SKILLS (50%): (required_skills matched / total required_skills) * 100.
2. EXPERIENCE (30%): years >= required → 100; 1yr short → 80;
   2yr short → 60; 3+yr short → 40; required is 0 → 100.
3. SENIORITY+LEADERSHIP (20%): seniority level matches → 60.
   Leadership: required & candidate has it → +40; required & missing → +0;
   not required → +40.

match_score = round(skills*0.5 + experience*0.3 + seniority_leadership*0.2)

OR CONDITIONS: having either option fully satisfies the requirement.
Never flag the unchosen alternative as missing.

FIELDS:
- matched_keywords: required_skills matched (literal, variant, or instance)
- missing_keywords: required_skills with no evidence anywhere — these are the
  actionable keywords to add. Exclude nice-to-have and OR alternatives.
- summary: 2-3 sentences — strengths, keyword gaps to add, readiness statement"""


# ── Core Function ─────────────────────────────────────────────────────────────
def _extract_job_requirements(jd_text: str) -> dict:
    """Extracts structured requirements from a job description."""

    structured_llm = llm_precise.with_structured_output(JobRequirements)

    messages = [
        SystemMessage(content=job_requirement_instructions),
        HumanMessage(content=jd_text)
    ]

    response = structured_llm.invoke(messages)
    return response.model_dump()


# ── StructuredTool ────────────────────────────────────────────────────────────
extract_job_requirements_tool = StructuredTool.from_function(
    func=_extract_job_requirements,
    name="extract_job_requirements",
    description="Extracts structured requirements from a job description including required skills, seniority level, OR conditions, and domain",
    args_schema=ExtractJobRequirementsInput
)


def _parse_cv(cv_text: str) -> dict:
    """Parses raw CV text into a structured candidate profile."""

    structured_llm = llm_precise.with_structured_output(CandidateProfile)

    messages = [
        SystemMessage(content=parse_cv_instructions),
        HumanMessage(content=cv_text)
    ]

    response = structured_llm.invoke(messages)
    return response.model_dump()


parse_cv_tool = StructuredTool.from_function(
    func=_parse_cv,
    name="parse_cv",
    description="Parses raw CV text into a structured candidate profile including skills, experience, education, certifications, and projects",
    args_schema=ParseCVInput
)


def _score_resume(profile: dict, requirements: dict) -> dict:
    """Scores a candidate profile against job requirements."""

    structured_llm = llm_precise.with_structured_output(ResumeScore)

    messages = [
        SystemMessage(content=score_rubric),
        HumanMessage(content=f"""Candidate Profile:
{json.dumps(profile, indent=2)}

Job Requirements:
{json.dumps(requirements, indent=2)}""")
    ]

    response = structured_llm.invoke(messages)
    return response.model_dump()

score_resume_tool = StructuredTool.from_function(
    func=_score_resume,
    name="score_resume",
    description="Scores a candidate profile against job requirements, returning a match score, matched and missing keywords, and a readiness summary",
    args_schema=ScoreResumeInput
)


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_jd = """
As a Sr. Data Scientist, you will serve as a lead on data science projects, collaborating with project/product managers, providing prioritization of tasks, balancing workload and mentoring data scientists on the project team. This role is expected to present insights and recommendations to leaders and business partners and explain the benefits and impacts of the recommended solutions. This role supports the building of skilled and talented data science teams by providing input to staffing needs and participating in the recruiting and hiring process. In addition, Data Scientists collaborate with business partners and cross-functional teams, requiring effective communication skills, building relationships and partnerships, and leveraging business proficiency to solutions and recommendations.

Key Responsibilities

35% Solution Development - Proficiently design and develop algorithms and models to use against large datasets to create business insights; Execute tasks with high levels of efficiency and quality; Make appropriate selection, utilization and interpretation of advanced analytical methodologies; Effectively communicate insights and recommendations to both technical and non-technical leaders and business customers/partners; Prepare reports, updates and/or presentations related to progress made on a project or solution; Clearly communicate impacts of recommendations to drive alignment and appropriate implementation
30% Project Management & Team Support - Work with project teams and business partners to determine project goals; Provide direction on prioritization of work and ensure quality of work; Provide mentoring and coaching to more junior roles to support their technical competencies; Collaborate with managers and team in the distribution of workload and resources; Support recruiting and hiring efforts for the team
20% Business Collaboration - Leverage extensive business knowledge into solution approach; Effectively develop trust and collaboration with internal customers and cross-functional teams; Provide general education on advanced analytics to technical and non-technical business partners; Deep understanding of IT needs for the team to be successful in tackling business problems; Actively seek out new business opportunities to leverage data science as a competitive advantage
15% Technical Exploration & Development - Seek further knowledge on key developments within data science, technical skill sets, and additional data sources; Participate in the continuous improvement of data science and analytics by developing replicable solutions (for example, codified data products, project documentation, process flowcharts) to ensure solutions are leveraged for future projects; Define best practices and develop clear vision for data analysis and model productionalization; Contribute to library of reusable algorithms for future use, ensuring developed codes are documented."""
    sample_cv = """
Vishnuventhira B R
Dallas, TX	E: vishnuventhira.br@outlook.com	 M: 860-294-2559	LinkedIn	Github
Senior Data Scientist with 9+ years’ experience in predictive modeling, risk analytics, and member lifetime value optimization. Led deployment of enterprise-scale production models across attrition, profitability, credit risk, churn prediction, and image classification impacting 14M+ members
EDUCATION
University of Connecticut School of Business, Hartford 					   Jan 2019 - May 2020
Master of Science in Business Analytics 								   GPA: 3.8/4.0
TECHNICAL SKILLS
Cloud & Infrastructure: AWS, Snowflake, Docker, Git, Linux, Databricks
Programming Languages: Python, R, SQL, PySpark
Machine Learning & AI: Supervised/Unsupervised Learning, Random Forest, XGBoost, Gradient Boosting, Deep Learning (CNN, LSTM), Time Series Forecasting, NLP, Computer Vision, Explainable AI (SHAP, LIME)
MLOps & Deployment: Model Training, Model Evaluation, Feature Engineering, A/B Testing, Experimentation, Model Monitoring
LLMs & Generative AI: RAG, Embeddings, Vector Databases, LangChain, LangGraph, Pydantic AI, Agentic AI
Statistics & Experimentation: Statistical Modeling, Hypothesis Testing, A/B Testing, Bayesian Methods, Survival Analysis, Regression, PCA, Regularization, Lift Analysis, RFM, Causal Inference 
Python Libraries: Pandas, NumPy, Scikit-learn, Statsmodels, Matplotlib, Seaborn, Plotly, TensorFlow, PyTorch, Keras, OpenCV, NLTK, SpaCy, Polars

PROFESSIONAL EXPERIENCE
Senior Data Scientist – USAA 								   Mar 2023 - Present
•	Architected and deployed production ML pipeline for member lifetime value (MLV) prediction across 7 product lines (Attrition, Deepening, Profitability), serving 14M+ members. Built scalable models using Python, Polars, and XGBoost, driving capital allocation and retention decisions across the enterprise.
•	Led development of regulatory compliance solution using CNN-LSTM hybrid model for check image classification (IRS/FinCEN Form 8300), achieving 98% recall and $5.2M annual cost savings. Integrated OCR (Tesseract) for automated keyword extraction and processed 12K images daily.
•	Engineered name normalization system using fuzzy matching (RapidFuzz) to standardize supplier names and P&C Auto Make-Model data, eliminating external vendor dependency. Solution achieved 95%+ matching accuracy across millions of records, enabling enterprise-wide spend analytics and reporting.
•	Collaborated with Product, Engineering, and Compliance teams to define model requirements, establish KPIs (Net Revenue, Expected Profit, Capital Allocation), and monitor production model performance using A/B testing frameworks.


Data Scientist II – First Tech Federal Credit Union 						June 2020 – Mar 2023
•	Built Member Lifetime Value (MLV) prediction system for 700K members using blended statistical approach combining RFM analysis, Survival Analysis, and cohort modeling in Python. Calculated Net Revenue forecasts across Auto Loans and Checking products to inform retention and cross-sell strategies.
•	Led Capital Planning and Stress Testing initiative, analyzing 10 years of loan performance data to forecast Probability of Default (PD), Loss Given Default (LGD), and Expected Loss (EL) for all portfolios. Fitted statistical models in R and performed sensitivity analysis using Federal Reserve stress scenarios to project portfolio behavior over 36 quarters, supporting regulatory compliance (CECL/CCAR)
•	Designed prepayment prediction model (binary classification) to identify mortgage members likely to refinance, enabling proactive retention campaigns. Model delivered $190K in saved interest revenue by retaining high-value 30-year fixed-rate auto loan members
•	Developed campaign response model using Logistic Regression to predict direct mail/email conversion probability, optimizing Personal Loan marketing spend and improving campaign ROI
System Engineer – Tata Consultancy Services 						 May 2015 – Nov 2018
•	Built customer churn prediction models (Logistic Regression, Decision Trees) analyzing 7M telecom customer records in Python, achieving 70% recall in top decile. Performed behavioral segmentation via cluster analysis to optimize marketing spend, reducing campaign costs by 30%.
•	Automated network configuration workflows using Ansible, creating reusable code libraries for switches and routers that eliminated 8 hours/week of manual deployment tasks.
ADDITIONAL PROJECTS
AI Data Analyst with Claude Code (Python, Claude Code, MCP)
Built agentic analytics system using Claude Code with multiple subagents, custom skills (markdown), and MCP server integration. Created automated data warehouse connector enabling natural language business intelligence queries.
Graduate Analytics Consultant – LIMRA | Insurance Customer Retention Model          Aug 2019 – Dec 2019
Developed Gradient Boosting model analyzing 9M insurance policies (70% accuracy) to predict policy lapses and identify key churn drivers. Created Tableau dashboards visualizing lapse patterns by gender, risk class, and distribution channel; presented recommendations to management improving agent profitability by 10%.

    """
    
    requirements = _extract_job_requirements(sample_jd)   # Tool 1 → requirements
    print("=== REQUIREMENTS ===")
    print(json.dumps(requirements, indent=2))

    profile = _parse_cv(sample_cv)                         # Tool 2 → profile
    print("\n=== CANDIDATE PROFILE ===")
    print(json.dumps(profile, indent=2))

    score = _score_resume(profile, requirements)           # Tool 3 → uses both
    print("\n=== SCORE ===")
    print(json.dumps(score, indent=2))