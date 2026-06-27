# 🤖 RedRob AI Candidate Ranker

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Gradio](https://img.shields.io/badge/Gradio-6.19-orange?logo=gradio)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Platform](https://img.shields.io/badge/HuggingFace-Spaces-yellow)

</p>

---

## 📌 Project Overview

**RedRob AI Candidate Ranker** is an AI-powered candidate ranking system developed for the **RedRob Senior AI Engineer Challenge**.

The application automatically analyzes candidate profiles from a JSONL dataset, evaluates each candidate using multiple scoring criteria, and generates an explainable ranking based on technical skills, professional experience, recruiter activity, semantic relevance, and other hiring signals.

Unlike traditional filtering systems, this project provides **transparent reasoning** for every ranked candidate, allowing recruiters to understand **why** a candidate received a particular score.

The project is deployed using **Hugging Face Spaces** with **Gradio** for an interactive web interface.

---

# 🌟 Features

✅ Intelligent AI Candidate Ranking

✅ Explainable AI Scoring

✅ Fast Processing of Large Candidate Datasets

✅ Recruiter-Friendly Dashboard

✅ Download Ranked CSV Submission

✅ Hugging Face Deployment

✅ JSONL & JSON Input Support

✅ Semantic Candidate Evaluation

✅ Offline Processing (No External API)

---

# 🚀 Live Demo

### Hugging Face Space

> https://huggingface.co/spaces/jigarm80/drkhack-redrob-ranker

---

# 🎯 Challenge Objective

The goal of this challenge is to build an intelligent candidate ranking engine capable of:

- Reading candidate profiles
- Understanding technical skills
- Evaluating recruiter interest
- Ranking candidates
- Providing human-readable reasoning
- Exporting submission CSV

---

# 🏗 System Architecture

```
                    +-------------------+
                    |  JSONL Candidates |
                    +---------+---------+
                              |
                              |
                    Read Candidate Data
                              |
                              ▼
                  +----------------------+
                  | Candidate Parser     |
                  +----------------------+
                              |
                              ▼
                  +----------------------+
                  | Feature Extraction   |
                  +----------------------+
                              |
                              ▼
             +--------------------------------+
             | AI Scoring Engine              |
             |                                |
             | Skills Score                   |
             | Experience Score               |
             | Semantic Matching              |
             | Recruiter Interest             |
             | Notice Period                  |
             | Education                      |
             | Job Stability                  |
             +--------------------------------+
                              |
                              ▼
                 Overall Candidate Score
                              |
                              ▼
                  Ranking & Sorting Engine
                              |
                              ▼
                    submission.csv Output
                              |
                              ▼
                     Gradio User Interface

```

---

# ⚙️ Project Workflow

```
Upload JSONL File
        │
        ▼
Read Candidate Profiles
        │
        ▼
Extract Features
        │
        ▼
Score Every Candidate
        │
        ▼
Generate Reasoning
        │
        ▼
Sort by Score
        │
        ▼
Generate submission.csv
        │
        ▼
Display Results
```

---

# 📂 Project Structure

```
redrob-ai-candidate-ranker/
│
├── app.py
├── rank_candidates.py
├── requirements.txt
├── runtime.txt
├── README.md
│
├── sample.jsonl
├── candidate_schema.json
├── validate_submission.py
├── submission_metadata_template.yaml
│
├── team_submission.csv
│
├── assets/
│   ├── homepage.png
│   ├── results.png
│   └── architecture.png
│
└── LICENSE
```

---

# 📄 File Description

## app.py

Creates the Gradio web application.

Responsibilities:

- Upload JSONL file
- Execute ranking engine
- Display ranked candidates
- Allow CSV download

---

## rank_candidates.py

Main AI ranking engine.

Responsibilities:

- Read candidate dataset
- Calculate candidate score
- Generate explanation
- Rank candidates
- Save submission.csv

---

## sample.jsonl

Sample candidate dataset for testing.

---

## validate_submission.py

Checks whether generated submission follows challenge specifications.

---

## candidate_schema.json

Defines the JSON schema expected by the ranking engine.

---

## requirements.txt

Contains all required Python dependencies.

---

## submission_metadata_template.yaml

Metadata required during challenge submission.

---

# 💻 Technology Stack

| Technology | Purpose |
|------------|----------|
| Python | Core Programming |
| Gradio | Web UI |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| JSONL | Candidate Dataset |
| Hugging Face Spaces | Deployment |
| Git | Version Control |
| GitHub | Source Code Hosting |

---

# 🔥 Key Highlights

- AI-based candidate ranking
- Explainable recommendations
- Human-readable reasoning
- Fast execution
- Hugging Face deployment
- Recruiter-friendly interface
- Open-source implementation

---

# 📊 Candidate Ranking Pipeline

```
Candidate Profile
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Experience Evaluation
        │
        ▼
Skill Matching
        │
        ▼
Recruiter Activity
        │
        ▼
Semantic Relevance
        │
        ▼
Final Score
        │
        ▼
Ranking
```
---

# 🛠 Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/jigar8800/redrob-ai-candidate-ranker.git
cd redrob-ai-candidate-ranker
```

---

## 2️⃣ Create a Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

Install all required Python packages.

```bash
pip install -r requirements.txt
```

Example **requirements.txt**

```text
gradio==6.19.0
pandas>=2.2.0
numpy>=2.0.0
```

---

## 4️⃣ Verify Installation

Check Python version

```bash
python --version
```

Expected

```
Python 3.13+
```

---

# 📦 Required Libraries

| Library | Purpose |
|----------|----------|
| Gradio | Web Interface |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| JSON | Read Candidate Files |
| CSV | Export Submission |
| argparse | CLI Support |
| subprocess | Execute Ranking Engine |
| tempfile | Temporary Files |
| shutil | File Management |
| os | Operating System Functions |
| sys | Python Runtime |

---

# ▶ Running the Project

## Option 1 — Run the Gradio Web App

```bash
python app.py
```

After running, Gradio will display something similar to

```
Running on local URL:

http://127.0.0.1:7860
```

Open the URL in your browser.

---

## Option 2 — Run the Ranking Engine from Terminal

```bash
python rank_candidates.py \
    --input sample.jsonl \
    --output submission.csv
```

This command reads the candidate dataset, ranks every candidate, and generates a CSV file.

---

# 🌐 Running on Hugging Face Spaces

The project is deployed using **Gradio** on **Hugging Face Spaces**.

Live Demo

```
https://huggingface.co/spaces/jigarm80/drkhack-redrob-ranker
```

Deploying on Hugging Face is simple.

Upload these files:

```
app.py
rank_candidates.py
requirements.txt
README.md
runtime.txt
sample.jsonl
```

After pushing your repository, Hugging Face automatically installs the dependencies from `requirements.txt` and launches the application.

---

# 📥 Input Format

The application accepts candidate profiles in **JSONL** format.

Example

```json
{
    "candidate_id":"CAND_000001",
    "name":"John Doe",
    "skills":[
        "Python",
        "Machine Learning",
        "NLP"
    ],
    "experience":6.5
}
```

Each line represents one complete candidate profile.

Example

```text
{"candidate_id":"CAND_001", ...}
{"candidate_id":"CAND_002", ...}
{"candidate_id":"CAND_003", ...}
```

---

# 📤 Output Format

The ranking engine generates

```
submission.csv
```

Example

| candidate_id | rank | score | reasoning |
|--------------|------|-------|-----------|
| CAND_000031 | 1 | 0.931 | Strong ML Engineer with production experience |
| CAND_000014 | 2 | 0.428 | Excellent semantic relevance and recruiter interest |
| CAND_000001 | 3 | 0.423 | Backend engineer with vector database experience |

---

# 🖥 Using the Web Application

## Step 1

Upload

```
sample.jsonl
```

---

## Step 2

Click

```
Run Ranking
```

---

## Step 3

Wait while the AI ranking engine evaluates every candidate.

---

## Step 4

View

- Candidate Rank
- Overall Score
- Reasoning
- Recruiter Recommendation

---

## Step 5

Download

```
submission.csv
```

---

# 📁 Example Repository Layout

```
redrob-ai-candidate-ranker
│
├── app.py
├── rank_candidates.py
├── README.md
├── requirements.txt
├── runtime.txt
│
├── sample.jsonl
├── submission.csv
├── candidate_schema.json
├── validate_submission.py
│
├── assets
│   ├── architecture.png
│   ├── homepage.png
│   └── ranking.png
│
└── LICENSE
```

---

# ⚡ Performance

Approximate execution time

| Candidates | Time |
|------------|------|
| 50 | < 1 second |
| 1,000 | 3–5 seconds |
| 10,000 | 30–60 seconds |
| 100,000 | Few minutes |

*(Performance depends on CPU and available memory.)*

---

# 📈 Generated Results

The application produces

- Ranked Candidate List
- Candidate Score
- Ranking Position
- Human-readable Reasoning
- Downloadable CSV

---

# 🧪 Testing

Use the included sample dataset.

```
sample.jsonl
```

Run

```bash
python app.py
```

Upload the sample file.

Expected output

```
Successfully ranked 50 candidates.
```

A ranked table should appear with scores and reasoning.

---

# 🐞 Troubleshooting

## "0 Candidates Ranked"

Ensure your input is in **JSONL** format.

Incorrect

```json
[
   { ... },
   { ... }
]
```

Correct

```text
{"candidate_id":"1", ...}
{"candidate_id":"2", ...}
```

---

## ModuleNotFoundError

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Hugging Face Shows Error

Check

```
Logs
```

inside your Hugging Face Space.

Most errors are related to

- Missing requirements
- Incorrect Python version
- Invalid JSONL
- Missing output CSV

---

## CSV Not Generated

Verify

- Input file exists
- JSONL format is valid
- `rank_candidates.py` executed successfully

---

---

# 🧠 AI Ranking Methodology

The ranking engine evaluates every candidate by analyzing multiple hiring signals instead of relying on a single metric.

Each candidate receives a final score computed from different weighted factors.

The overall objective is to rank candidates based on their likelihood of being a strong fit for an AI Engineering role.

---

# 🎯 Scoring Factors

The ranking engine considers the following information.

| Feature | Description |
|----------|-------------|
| Technical Skills | AI, ML, NLP, Python, Vector Databases, Search Technologies |
| Years of Experience | Professional experience in software engineering |
| Current Role | Relevance of current job title |
| Previous Companies | Industry experience |
| Recruiter Activity | Recruiter saves, profile views, shortlists |
| Semantic Relevance | Candidate profile similarity |
| Education | Degree and academic background |
| Job Stability | Employment history |
| Notice Period | Candidate availability |
| Open To Work | Immediate availability |

---

# 📊 Overall Score

The final score is calculated by combining multiple weighted signals.

```
Overall Score =
Technical Skills
+ Experience
+ Semantic Match
+ Recruiter Interest
+ Education
+ Company Experience
- Penalties
```

Candidates are then sorted in descending order.

---

# 🔎 Feature Engineering

Before ranking, the dataset undergoes preprocessing.

Processing includes:

- Missing value handling
- Text normalization
- Skill extraction
- Experience normalization
- Recruiter signal extraction
- Notice period conversion
- Duplicate removal
- Semantic keyword matching

This improves ranking quality and consistency.

---

# 🤖 Explainable AI

Unlike traditional ranking systems, every ranked candidate includes a human-readable explanation.

Example

```
Recommendation Systems Engineer with 6 years of production experience.
Strong expertise in Embeddings and Pinecone.
High recruiter engagement.
Notice period is acceptable.
```

This allows recruiters to understand why a candidate achieved a particular rank.

---

# 📈 Candidate Ranking Flow

```
Candidate Profile
        │
        ▼
Preprocessing
        │
        ▼
Feature Extraction
        │
        ▼
Weighted Scoring
        │
        ▼
Reason Generation
        │
        ▼
Sorting
        │
        ▼
CSV Export
```

---

# 📊 Example Ranking

| Rank | Candidate | Score |
|------|-----------|-------|
| 1 | Recommendation Systems Engineer | 0.931 |
| 2 | Applied ML Engineer | 0.427 |
| 3 | Backend Engineer | 0.423 |
| 4 | Senior Data Engineer | 0.381 |
| 5 | Frontend Engineer | 0.374 |

---

# 🎯 Evaluation Strategy

The ranking quality is evaluated based on:

- Skill relevance
- Candidate completeness
- Recruiter engagement
- Professional experience
- Explainability
- Overall ranking consistency

---

# 🚀 Why This Project?

Hiring thousands of candidates manually is slow and subjective.

This project demonstrates how AI can assist recruiters by:

- Reducing manual effort
- Improving candidate prioritization
- Providing transparent rankings
- Supporting faster hiring decisions

---

# 🌍 Real-World Applications

This ranking system can be adapted for:

- Recruitment platforms
- HR management systems
- Resume screening
- Talent acquisition
- Internal employee promotion
- University placement portals
- Freelance marketplaces

---

# 🔒 Privacy & Security

This application processes candidate data locally.

Features:

- No external API calls
- No cloud-based AI inference
- No candidate information is stored permanently
- Temporary files are removed after processing (if enabled)

This helps protect sensitive applicant information.

---

# ⚙️ Performance Optimization

The project is optimized using:

- Efficient JSON parsing
- Lightweight preprocessing
- Vectorized data handling with Pandas
- Fast sorting algorithms
- Minimal memory usage
- CPU-friendly execution

---

# 🧩 Future Improvements

Planned enhancements include:

- Resume PDF parsing
- LLM-powered candidate summaries
- Semantic embeddings
- FAISS similarity search
- Pinecone vector database integration
- Skill ontology matching
- Recruiter dashboard
- Analytics dashboard
- Candidate filtering
- REST API support
- Docker deployment
- Kubernetes deployment
- Multi-language support

---

# 📸 Demo Screenshots

## Upload Page

```
assets/upload-page.png
```

---

## Ranking Results

```
assets/ranking-results.png
```

---

## Download CSV

```
assets/download-results.png
```

---

## Hugging Face Space

```
assets/huggingface-space.png
```

---

# 📚 References

- Python Documentation
- Gradio Documentation
- Hugging Face Spaces
- Pandas Documentation
- NumPy Documentation
- JSON Lines Specification

---

# 🏆 Project Highlights

✔ AI-powered candidate ranking

✔ Explainable scoring

✔ Recruiter-friendly interface

✔ Fast execution

✔ Open-source implementation

✔ Hugging Face deployment

✔ JSONL support

✔ Downloadable submission

✔ Production-ready structure

---

---

# 🛣️ Project Roadmap

The following features are planned for future releases.

## Version 1.1

- Improved candidate ranking algorithm
- Better explainable reasoning
- Enhanced recruiter dashboard
- Candidate search functionality

---

## Version 1.2

- Resume PDF Upload
- Automatic Resume Parsing
- Job Description Matching
- Candidate Recommendation Engine

---

## Version 2.0

- Large Language Model Integration
- Semantic Resume Search
- Vector Database Support
- AI Interview Question Generator
- Recruiter Analytics Dashboard
- Candidate Skill Gap Analysis

---

# 🤝 Contributing

Contributions are always welcome!

If you'd like to improve this project:

1. Fork this repository.
2. Create a new feature branch.

```bash
git checkout -b feature/my-feature
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push your branch.

```bash
git push origin feature/my-feature
```

5. Open a Pull Request.

Please ensure that your code follows clean coding practices and is well documented.

---

# 📝 Coding Standards

This project follows:

- PEP 8 Style Guide
- Meaningful variable names
- Modular architecture
- Readable documentation
- Consistent formatting

---

# 🧪 Testing Checklist

Before submitting any changes, verify:

- Application launches successfully
- JSONL files are parsed correctly
- Candidate ranking is generated
- CSV export works
- Hugging Face deployment runs successfully
- No Python exceptions are raised

---

# 📦 Deployment

This project is deployed using **Hugging Face Spaces**.

Deployment includes:

- Python 3.13
- Gradio
- Pandas
- NumPy

Required files:

```
app.py
rank_candidates.py
requirements.txt
README.md
runtime.txt
```

---

# 📈 Project Statistics

| Metric | Value |
|---------|------:|
| Programming Language | Python |
| Framework | Gradio |
| Deployment | Hugging Face Spaces |
| Dataset Format | JSONL |
| Output Format | CSV |
| Open Source | Yes |
| Explainable AI | Yes |

---

# ❓ Frequently Asked Questions

### Does this project require an API key?

No.

The application runs completely offline.

---

### Does it use OpenAI or Gemini?

No.

The ranking engine works locally without external AI APIs.

---

### Can I use my own candidate dataset?

Yes.

Provide a JSONL file that matches the expected schema.

---

### Does it support large datasets?

Yes.

The ranking engine is designed to process large candidate datasets efficiently.

---

### Can I deploy it locally?

Yes.

Simply install the requirements and run:

```bash
python app.py
```

---

# 📜 License

This project is released under the **MIT License**.

You are free to:

- Use
- Modify
- Distribute
- Learn from the project

Please retain the original copyright notice.

See the `LICENSE` file for complete details.

---

# 🙏 Acknowledgements

Special thanks to:

- RedRob AI Challenge organizers
- Hugging Face
- Gradio
- Python Community
- Pandas Development Team
- NumPy Development Team
- Open Source Community

---

# 👨‍💻 Author

**Jigar M.**

🎓 Bachelor of Computer Applications (BCA)

💻 Python Developer | AI & Machine Learning Enthusiast

🌐 GitHub:
https://github.com/jigar8800

🤗 Hugging Face:
https://huggingface.co/jigarm80

---

# 📬 Contact

For questions, suggestions, or collaboration:

📧 Email:
jigarmaturkar31@gmail.com



---

# ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork the repository

🛠️ Contribute improvements

📢 Share it with others

Your support helps improve the project and encourages future development.

---

# 📌 Repository

```
GitHub
https://github.com/jigar8800/redrob-ai-candidate-ranker
```

---

# 🚀 Live Demo

```
Hugging Face Space
https://github.com/Jigar8800/redrob-ai-candidate-ranker.git
```

---

# 📊 Current Status

✅ Project Completed

✅ Ranking Engine Implemented

✅ Explainable AI Scoring

✅ Hugging Face Deployment

✅ Recruiter-Friendly Interface

✅ CSV Export

✅ Open Source

---

# 🌟 If you like this project...

Please consider giving it a ⭐ on GitHub!

It helps others discover the project and motivates continued development.

---

<p align="center">

## 🚀 Thank You for Visiting!

**Made with ❤️ using Python, Gradio, and Hugging Face Spaces**

</p>
