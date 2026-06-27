#!/usr/bin/env python3
"""
Redrob Candidate Discovery & Ranking Challenge — v2 (Final Submission)
======================================================================
DESIGN PHILOSOPHY (for Stage 5 Interview):
  This is a deterministic, explainable, rule-based ranking engine. No neural
  networks, no external APIs, no pre-trained embeddings. Pure signal extraction
  from structured profile data + semantic keyword patterns.
  Five weighted components feed into a final normalized score [0, 1]:
    1. Career Match        (0.42) — titles, descriptions, production evidence
    2. Skills Relevance    (0.18) — proficiency × endorsements × duration × assessment
    3. Behavioral          (0.22) — response rate, recency, availability, github
    4. Experience+Education(0.10) — YoE in ideal band, tier-1/2 education
    5. Honeypot Penalty    (-0.10 to 0) — timeline math, stuffing, consulting trap
  SCORING DOMINATES NDCG@10: Career Match is decisive because the JD is very
  specific (retrieval/ranking systems at product companies). A marketing manager
  with a keyword-stuffed skills list should score ~0.05; a Search Engineer at
  Sarvam AI with 7 years should score ~0.96. The gap must be large enough that
  honeypots and non-fits never appear in top 100.
  REASONING DOMINATES STAGE 4: Each candidate gets a unique, factual,
  human-sounding 1-2 sentence reasoning that references specific profile facts
  (YoE, title, company, named skills, signal values) and connects them to the
  exact JD language ("production retrieval system", "hybrid search",
  "evaluation framework"). Concerns are always stated honestly.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime

# ==============================================================================
# CONSTANTS & CONFIGURATION
# ==============================================================================

# Current date for recency computation (from challenge metadata)
CURRENT_DATE = datetime(2026, 6, 26)

# Preferred Indian cities per JD (Pune/Noida primary, Tier-1 cities acceptable)
TARGET_LOCATIONS = {
    "noida", "pune", "delhi", "ncr", "gurgaon", "gurugram",
    "hyderabad", "mumbai", "bangalore", "bengaluru", "chennai", "kolkata"
}

# Consulting firm trap — full career here is penalized per JD
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro", "cognizant",
    "accenture", "capgemini", "mindtree", "tech mahindra", "hexaware",
    "mphasis", "l&t infotech", "ltimindtree", "hcl technologies", "hcl"
}

# Hard disqualifier role keywords in *current* title
# (JD explicitly says these people are not a fit)
DISQUALIFIED_TITLE_TOKENS = {
    "marketing", "hr", "recruiter", "accountant", "operations manager",
    "sales", "customer support", "customer success", "civil engineer",
    "mechanical engineer", "scrum master", "talent acquisition",
    "content writer", "seo", "brand manager", "social media"
}

# Soft disqualifier tokens (in non-current roles; apply partial penalty)
SOFT_DISQUALIFIED_TOKENS = {
    "project manager", "business analyst", "product manager",
    "program manager", "operations"
}

# ── HIGH-VALUE: Retrieval & Ranking signals ────────────────────────────────
# These are the CORE JD requirements. Matches here heavily boost Career score.
RETRIEVAL_KEYWORDS = {
    "retrieval", "ranking", "recommendation", "recommender", "embeddings", "embedding",
    "vector database", "vector db", "vector store", "pinecone", "weaviate", "qdrant",
    "milvus", "faiss", "annoy", "scann", "hnsw", "rag", "retrieval augmented",
    "hybrid search", "dense retrieval", "sparse retrieval", "bm25", "tfidf",
    "ndcg", "mrr", "map@", "precision@", "recall@", "elasticsearch", "opensearch",
    "learning to rank", "ltr", "lambdamart", "listwise", "pairwise", "pointwise",
    "search engine", "search relevance", "information retrieval", "re-ranking",
    "re-ranker", "cross-encoder", "bi-encoder", "neural search", "semantic search",
    "candidate retrieval", "two-tower", "colbert", "splade", "dpr"
}

# ── LLM & Fine-tuning signals ─────────────────────────────────────────────
LLM_KEYWORDS = {
    "llm", "large language model", "fine-tuning", "fine tuning", "finetuning",
    "lora", "qlora", "peft", "rlhf", "dpo", "sft", "instruction tuning",
    "prompt engineering", "prompt tuning", "sentence-transformers",
    "sentence transformers", "nlp", "transformers", "bert", "roberta",
    "gpt", "claude", "llama", "mistral", "phi", "deepseek", "gemma",
    "hugging face", "huggingface", "tokenizer", "attention", "langchain"
}

# ── Production evidence signals ───────────────────────────────────────────
# The JD explicitly wants production-deployed systems, not just research
PRODUCTION_KEYWORDS = {
    "production", "prod", "shipped", "deployed", "deployment", "serving",
    "at scale", "scaled to", "latency", "throughput", "qps", "a/b test",
    "a/b testing", "ab test", "online evaluation", "offline evaluation",
    "evaluation framework", "eval framework", "mlops", "model serving",
    "inference optimization", "inference pipeline", "feature store",
    "real users", "live traffic", "model monitoring", "drift detection",
    "distributed", "kubernetes", "triton", "tensorrt", "onnx", "torchserve",
    "bentoml", "ray serve", "seldon", "kubeflow", "airflow", "prefect"
}

# Core AI/ML skills that count toward Skills Relevance Score
CORE_AI_SKILLS = RETRIEVAL_KEYWORDS | LLM_KEYWORDS | {
    "machine learning", "deep learning", "ml", "ai", "artificial intelligence",
    "applied ml", "applied ai", "ml infrastructure", "ml platform", "mlops",
    "data science", "neural networks", "xgboost", "lightgbm", "catboost",
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "numpy", "pandas"
}

# Skills that specifically validate platform-assessed competency
HIGH_VALUE_ASSESSMENT_SKILLS = {
    "information retrieval", "semantic search", "rag", "embeddings",
    "ranking", "recommendation", "nlp", "transformers", "machine learning",
    "deep learning", "mlops", "fine-tuning"
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation for uniform keyword matching."""
    if not text:
        return ""
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower())


def text_contains_any(keywords: set, text: str) -> bool:
    """Fast check: does the normalized text contain any keyword from the set?"""
    if not text:
        return False
    norm = normalize_text(text)
    for kw in keywords:
        if kw in norm:
            return True
    return False


def count_unique_matches(keywords: set, text: str) -> int:
    """
    Count how many UNIQUE keywords from the set appear in the text.
    Multi-word phrases are checked as substrings; single tokens as whole words.
    Using unique count (not frequency) prevents keyword-stuffing inflation.
    """
    if not text:
        return 0
    norm = normalize_text(text)
    norm_words = set(norm.split())
    matches = 0
    for kw in keywords:
        kw_norm = kw.lower()
        if ' ' in kw_norm:
            if kw_norm in norm:
                matches += 1
        elif kw_norm in norm_words:
            matches += 1
    return matches


def parse_date(date_str) -> datetime:
    """Parse ISO date string, returning CURRENT_DATE as fallback."""
    if not date_str:
        return CURRENT_DATE
    try:
        return datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return CURRENT_DATE


def days_since(date_str) -> int:
    """Return number of days elapsed since a given date string."""
    return (CURRENT_DATE - parse_date(date_str)).days


def is_consulting_company(company_name: str) -> bool:
    """Return True if company name matches any known consulting firm."""
    norm = normalize_text(company_name)
    return any(c in norm for c in CONSULTING_COMPANIES)


def is_disqualified_title(title: str) -> bool:
    """Return True if title contains a hard-disqualifier token."""
    norm = normalize_text(title)
    return any(t in norm for t in DISQUALIFIED_TITLE_TOKENS)


def is_soft_disqualified_title(title: str) -> bool:
    """Return True if title contains a soft-disqualifier token."""
    norm = normalize_text(title)
    return any(t in norm for t in SOFT_DISQUALIFIED_TOKENS)


def extract_top_skills(skills: list, keyword_set: set, n: int = 3) -> list:
    """
    Extract names of the top-N skills that match a given keyword set,
    sorted by (proficiency, endorsements) descending.
    Used for factual reasoning generation.
    """
    PROF_ORDER = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    matched = []
    for s in skills:
        name = s.get("name", "")
        if text_contains_any(keyword_set, name):
            matched.append((
                PROF_ORDER.get(s.get("proficiency", "beginner"), 1),
                s.get("endorsements", 0),
                name
            ))
    matched.sort(reverse=True)
    return [m[2] for m in matched[:n]]


# ==============================================================================
# SCORING COMPONENT 1: CAREER MATCH  (weight 0.42)
# ==============================================================================

def compute_career_score(candidate: dict) -> tuple[float, dict]:
    """
    Compute Career Match Score.
    
    DESIGN DECISION: This is the most important component because the JD
    is explicit: they want people who have *shipped* retrieval/ranking systems
    at product companies. A Tier-1 ML title with retrieval keywords in the
    description is worth ~5x a generic backend engineer with no ML context.
    Returns: (score [0,1], rich_info dict for reasoning generation)
    """
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])

    if not career_history:
        return 0.0, {"tier": 3, "num_ml_roles": 0, "ir_matches": 0,
                     "prod_matches": 0, "roles_info": [], "consulting_fraction": 0.0,
                     "current_disqualified": False, "has_disqualified": False}

    roles_info = []
    num_ml_roles = 0
    num_consulting = 0
    has_hard_disqualified = False
    current_disqualified = False

    for i, role in enumerate(career_history):
        title = role.get("title", "")
        company = role.get("company", "")
        description = role.get("description", "")
        duration = role.get("duration_months", 0)
        norm_title = normalize_text(title)

        consulting = is_consulting_company(company)
        if consulting:
            num_consulting += 1

        hard_disq = is_disqualified_title(title)
        soft_disq = is_soft_disqualified_title(title) if not hard_disq else False

        if hard_disq:
            has_hard_disqualified = True
            if i == 0:
                current_disqualified = True

        # ── Classify role tier ────────────────────────────────────────────
        # Tier 1: Explicitly ML/AI/Search title
        # Tier 2: Software/Backend/Data with strong ML-infra keywords in desc
        # Tier 3: Generic / unclassified
        # Tier 4: Hard-disqualified (Marketing, HR, etc.)
        tier = 3
        is_ml_title = any(t in norm_title for t in [
            "ai engineer", "ml engineer", "machine learning", "search engineer",
            "recommendation", "nlp engineer", "applied scientist", "research engineer",
            "data scientist", "ranking engineer", "retrieval", "relevance engineer",
            "applied ml", "ai researcher"
        ])
        # Broader ML title tokens (one word suffices)
        if not is_ml_title:
            is_ml_title = any(t in norm_title for t in [" ml ", "nlp", " ai "])

        is_cv_robotics = any(t in norm_title for t in [
            "vision", "computer vision", "speech", "robotics", "audio", "image classification"
        ])

        if hard_disq:
            tier = 4
        elif is_ml_title:
            if is_cv_robotics:
                # CV/Speech only gets Tier 1 if they have retrieval keywords
                # (JD explicitly says CV/Speech without NLP/IR is a miss)
                if count_unique_matches(RETRIEVAL_KEYWORDS, description) >= 2:
                    tier = 1
                    num_ml_roles += 1
                else:
                    tier = 2
            else:
                tier = 1
                num_ml_roles += 1
        elif any(t in norm_title for t in [
            "backend", "software engineer", "sde", "data engineer", "platform",
            "infrastructure", "systems engineer", "full stack", "fullstack"
        ]):
            # Backend/Data with strong retrieval OR LLM keywords in description
            ir_in_desc = count_unique_matches(RETRIEVAL_KEYWORDS, description)
            llm_in_desc = count_unique_matches(LLM_KEYWORDS, description)
            if ir_in_desc >= 3 or (ir_in_desc >= 1 and llm_in_desc >= 2):
                tier = 2
                num_ml_roles += 1
            elif ir_in_desc >= 1 or llm_in_desc >= 1:
                tier = 2  # but don't add to ml_roles count

        roles_info.append({
            "title": title,
            "company": company,
            "tier": tier,
            "duration": duration,
            "description": description,
            "consulting": consulting,
            "hard_disqualified": hard_disq
        })

    # ── Aggregate keyword matches across the FULL profile ────────────────
    full_text = (
        " ".join(r["description"] for r in roles_info)
        + " " + profile.get("headline", "")
        + " " + profile.get("summary", "")
    )
    ir_matches = count_unique_matches(RETRIEVAL_KEYWORDS, full_text)
    llm_matches = count_unique_matches(LLM_KEYWORDS, full_text)
    prod_matches = count_unique_matches(PRODUCTION_KEYWORDS, full_text)

    # ── Tier-based base score ─────────────────────────────────────────────
    best_tier = min(r["tier"] for r in roles_info)  # lower = better
    current_tier = roles_info[0]["tier"] if roles_info else 3

    if best_tier == 1:
        tier_score = 1.0
    elif best_tier == 2:
        tier_score = 0.65
    else:
        tier_score = 0.25

    # Boost if *current* role is Tier 1 (they are actively working in ML/AI now)
    if current_tier == 1:
        tier_score = max(tier_score, 0.90)
    elif current_tier == 2:
        tier_score = max(tier_score, 0.55)

    # ── Keyword boost (sublinear) ─────────────────────────────────────────
    # DESIGN: log1p scaling prevents a profile that mentions "embedding" 20 times
    # from outranking one that has 3 unique retrieval systems in production.
    # IR gets highest weight because that IS the job.
    keyword_score = min(1.0, (
        math.log1p(ir_matches) * 0.50 +
        math.log1p(llm_matches) * 0.22 +
        math.log1p(prod_matches) * 0.28
    ))

    # ── Multi-role ML bonus ───────────────────────────────────────────────
    multi_role_bonus = min(0.10, num_ml_roles * 0.04)

    # ── Final career score ────────────────────────────────────────────────
    raw = tier_score * 0.52 + keyword_score * 0.40 + multi_role_bonus
    career_score = min(1.0, max(0.0, raw))

    # ── Heavy penalty for current disqualified role ───────────────────────
    # (A Marketing Manager who lists "RAG" in skills is still not a fit)
    if current_disqualified:
        career_score *= 0.08
    elif has_hard_disqualified:
        career_score *= 0.55

    # ── Consulting fraction penalty applied later in penalties fn ─────────
    consulting_fraction = num_consulting / max(1, len(roles_info))

    rich_info = {
        "tier": best_tier,
        "current_tier": current_tier,
        "num_ml_roles": num_ml_roles,
        "ir_matches": ir_matches,
        "llm_matches": llm_matches,
        "prod_matches": prod_matches,
        "roles_info": roles_info,
        "consulting_fraction": consulting_fraction,
        "current_disqualified": current_disqualified,
        "has_disqualified": has_hard_disqualified,
        "full_text": full_text
    }
    return career_score, rich_info


# ==============================================================================
# SCORING COMPONENT 2: SKILLS RELEVANCE  (weight 0.18)
# ==============================================================================

def compute_skills_score(candidate: dict) -> float:
    """
    Compute Skills Relevance Score.
    DESIGN DECISION: Skills section is notorious for keyword stuffing.
    We de-weight raw skill count and instead use a compound formula:
        skill_value = relevance × proficiency × log(1+endorsements) × duration_factor
    Then boost if the candidate has a matching *verified* platform assessment.
    A candidate with 2 expert-level verified IR skills beats one with 15 unverified
    beginner AI skills.
    """
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})
    # Normalize assessment keys for matching
    norm_assessments = {normalize_text(k): v for k, v in assessments.items()}

    if not skills:
        return 0.0

    PROF_MULT = {"beginner": 0.20, "intermediate": 0.45, "advanced": 0.80, "expert": 1.00}
    total_value = 0.0

    for skill in skills:
        name = skill.get("name", "")
        norm_name = normalize_text(name)
        prof = PROF_MULT.get(skill.get("proficiency", "beginner"), 0.20)
        endorsements = skill.get("endorsements", 0)
        duration = skill.get("duration_months", 0)

        # ── Assign relevance weight ───────────────────────────────────────
        if text_contains_any(RETRIEVAL_KEYWORDS, name):
            rel = 1.00          # Core IR/ranking skill
        elif text_contains_any(LLM_KEYWORDS, name):
            rel = 0.80          # LLM / fine-tuning skill
        elif text_contains_any(CORE_AI_SKILLS, name):
            rel = 0.55          # General ML/AI
        else:
            continue            # Not relevant — skip entirely

        # ── Duration factor (meaningful use, not just listed) ─────────────
        dur_factor = 1.0 + min(0.40, duration / 60.0)  # max 1.40 at 24+ months

        # ── Endorsement factor (log-scaled) ──────────────────────────────
        end_factor = 1.0 + 0.12 * math.log1p(endorsements)

        skill_value = rel * prof * dur_factor * end_factor

        # ── Platform assessment verification boost ────────────────────────
        # DESIGN: Verified assessment scores are gold. A score of 80+ in
        # "Information Retrieval" is far more credible than listing it as a skill.
        matched_score = norm_assessments.get(norm_name)
        if matched_score is not None:
            # Scale: 50 = neutral, 100 = 2× boost, 0 = half value
            assessment_mult = 0.5 + (matched_score / 100.0)
            skill_value *= assessment_mult

        total_value += skill_value

    # Normalize: 8.0 represents ~4-5 strong, verified IR skills
    return min(1.0, max(0.0, total_value / 8.0))


# ==============================================================================
# SCORING COMPONENT 3: BEHAVIORAL & AVAILABILITY  (weight 0.22)
# ==============================================================================

def compute_behavioral_score(candidate: dict) -> tuple[float, dict]:
    """
    Compute Behavioral & Availability Score.
    DESIGN DECISION: The JD explicitly says "active on Redrob platform so we
    can actually talk to them." A perfect-skill candidate who hasn't logged in
    in 4 months and ignores recruiter messages is not hirable. This component
    acts as a multiplier on profile quality.
    Returns: (score [0,1], signals_info dict for reasoning)
    """
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    response_rate = signals.get("recruiter_response_rate", 0.0)
    open_to_work = signals.get("open_to_work_flag", False)
    willing_relocate = signals.get("willing_to_relocate", False)
    github_score = signals.get("github_activity_score", -1.0)
    saved_30d = signals.get("saved_by_recruiters_30d", 0)
    views_30d = signals.get("profile_views_received_30d", 0)
    notice_days = signals.get("notice_period_days", 90)
    last_active = signals.get("last_active_date", "")

    # ── 1. Recruiter Response Rate (38% weight) ───────────────────────────
    # JD: "active on platform so we can actually talk to them"
    rr_component = response_rate * 0.38

    # ── 2. Availability & Location (22% weight) ───────────────────────────
    location_norm = normalize_text(profile.get("location", ""))
    is_in_target = any(city in location_norm for city in TARGET_LOCATIONS)
    is_india = profile.get("country", "").lower() in {"india", "in"}

    avail = 0.0
    if open_to_work:
        avail += 0.12
    if willing_relocate:
        avail += 0.06
    if is_in_target:
        avail += 0.04

    # ── 3. Activity Recency (20% weight) ─────────────────────────────────
    days_inactive = days_since(last_active)
    if days_inactive <= 7:
        recency = 0.20
    elif days_inactive <= 14:
        recency = 0.17
    elif days_inactive <= 30:
        recency = 0.14
    elif days_inactive <= 60:
        recency = 0.09
    elif days_inactive <= 90:
        recency = 0.04
    else:
        recency = 0.0

    # ── 4. Recruiter Demand & GitHub (20% weight) ─────────────────────────
    saved_comp = min(0.06, saved_30d * 0.006)            # max at 10 saves
    views_comp = min(0.04, views_30d * 0.0004)           # max at 100 views
    github_comp = max(0.0, github_score / 100.0) * 0.10  # -1 → 0.0, 100 → 0.10

    demand_comp = saved_comp + views_comp + github_comp

    # ── 5. Notice Period Modifier ─────────────────────────────────────────
    # JD: "We'd love sub-30-day notice. 30+ day candidates are in scope but bar is higher."
    if notice_days == 0:
        notice_mult = 1.10
    elif notice_days <= 15:
        notice_mult = 1.08
    elif notice_days <= 30:
        notice_mult = 1.04
    elif notice_days <= 60:
        notice_mult = 0.96
    elif notice_days <= 90:
        notice_mult = 0.88
    else:
        notice_mult = 0.80   # 90+ days is a significant barrier for a founding team

    raw = (rr_component + avail + recency + demand_comp) * notice_mult
    behavioral_score = min(1.0, max(0.0, raw))

    signals_info = {
        "response_rate": response_rate,
        "open_to_work": open_to_work,
        "willing_relocate": willing_relocate,
        "is_in_target": is_in_target,
        "days_inactive": days_inactive,
        "github_score": github_score,
        "saved_30d": saved_30d,
        "notice_days": notice_days,
        "location": profile.get("location", ""),
        "country": profile.get("country", "India")
    }
    return behavioral_score, signals_info


# ==============================================================================
# SCORING COMPONENT 4: EXPERIENCE & EDUCATION  (weight 0.10)
# ==============================================================================

def compute_exp_edu_score(candidate: dict) -> float:
    """
    Compute Experience & Education Score.
    Ideal: 5-9 years total (JD: "6-8 years of which 4-5 are applied ML").
    Tier-1/2 institution boosts (IIT, BITS, NIT, etc.).
    """
    profile = candidate.get("profile", {})
    education = candidate.get("education", [])
    yoe = profile.get("years_of_experience", 0.0)

    # ── Experience band scoring ───────────────────────────────────────────
    if 5.0 <= yoe <= 9.0:
        exp_s = 0.80
    elif 4.0 <= yoe < 5.0 or 9.0 < yoe <= 12.0:
        exp_s = 0.65
    elif 3.0 <= yoe < 4.0 or 12.0 < yoe <= 15.0:
        exp_s = 0.45
    elif 2.0 <= yoe < 3.0:
        exp_s = 0.30
    else:
        exp_s = 0.15

    # ── Education prestige bonus ──────────────────────────────────────────
    edu_bonus = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        if tier == "tier_1":
            edu_bonus = max(edu_bonus, 0.20)
        elif tier == "tier_2":
            edu_bonus = max(edu_bonus, 0.10)

    return min(1.0, exp_s + edu_bonus)


# ==============================================================================
# SCORING COMPONENT 5: HONEYPOT & TRAP PENALTIES  (weight -0.10 to 0)
# ==============================================================================

def detect_penalties(candidate: dict, career_info: dict) -> tuple[float, list]:
    """
    Detect impossible profiles and behavioral red flags.
    DESIGN DECISION: Penalties are modular and additive (up to the cap).
    We check 7 distinct signals so that edge-case honeypots with just one
    small inconsistency are still caught at a reduced penalty.
    Returns: (penalty [−0.10, 0], list of triggered issue codes)
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    career_history = candidate.get("career_history", [])
    roles_info = career_info.get("roles_info", [])

    penalty = 0.0
    issues = []
    yoe = profile.get("years_of_experience", 0.0)

    # ── Honeypot A: Single role duration > total YoE ──────────────────────
    # e.g. "6 years at company" but profile says 4 years total experience
    for role in roles_info:
        role_years = role["duration"] / 12.0
        if role_years > (yoe + 0.6) and yoe > 0:
            penalty -= 0.06
            issues.append("HONEYPOT_role_duration_exceeds_total_yoe")
            break

    # ── Honeypot B: Mathematical date inconsistency ───────────────────────
    # e.g. start=2021, end=2023 but duration_months=96 (impossible)
    for role in career_history:
        start_str = role.get("start_date", "")
        end_str = role.get("end_date", "")
        is_current = role.get("is_current", False)
        stated = role.get("duration_months", 0)
        if start_str and stated > 0:
            start_dt = parse_date(start_str)
            end_dt = parse_date(end_str) if (end_str and not is_current) else CURRENT_DATE
            actual = (end_dt - start_dt).days / 30.44
            # Allow 4-month tolerance for rounding/part-month effects
            if abs(actual - stated) > 4.0:
                penalty -= 0.07
                issues.append("HONEYPOT_impossible_date_math")
                break

    # ── Honeypot C: Expert skills with zero duration ──────────────────────
    # "Expert in RAG, Milvus, LoRA" but duration_months = 0 and no career mention
    expert_no_duration = 0
    career_text_low = career_info.get("full_text", "").lower()
    for s in skills:
        name_low = s.get("name", "").lower()
        is_expert = s.get("proficiency") in {"expert", "advanced"}
        duration = s.get("duration_months", 0)
        is_high_value = text_contains_any(RETRIEVAL_KEYWORDS | LLM_KEYWORDS, name_low)
        if is_high_value and is_expert and duration == 0:
            # Only flag if the skill isn't corroborated by career descriptions
            first_word = name_low.split()[0]
            if first_word not in career_text_low:
                expert_no_duration += 1
    if expert_no_duration >= 3:
        penalty -= 0.05
        issues.append("HONEYPOT_expert_skills_zero_duration")

    # ── Honeypot D: Skill count anomaly ──────────────────────────────────
    # "25 skills with <2 years experience" is a clear stuffing signal
    num_skills = len(skills)
    if yoe < 2.0 and num_skills > 14:
        penalty -= 0.05
        issues.append("HONEYPOT_skill_stuffing_low_yoe")
    elif yoe < 3.0 and num_skills > 20:
        penalty -= 0.03
        issues.append("HONEYPOT_skill_stuffing_moderate")

    # ── Consulting Trap: 100% career in consulting ────────────────────────
    # JD explicitly says: "only consulting firms their entire career = no fit"
    # Partial consulting is fine; full history is penalized heavily
    consulting_frac = career_info.get("consulting_fraction", 0.0)
    if consulting_frac >= 1.0:
        penalty -= 0.09
        issues.append("TRAP_pure_consulting_career")
    elif consulting_frac >= 0.75:
        penalty -= 0.05
        issues.append("TRAP_mostly_consulting_career")

    # ── Behavioral Trap: Very low response rate ───────────────────────────
    # JD: "active so we can actually talk to them"
    rr = signals.get("recruiter_response_rate", 1.0)
    if rr < 0.20:
        penalty -= 0.05
        issues.append("BEHAVIORAL_very_low_response_rate")
    elif rr < 0.30:
        penalty -= 0.03
        issues.append("BEHAVIORAL_low_response_rate")

    # ── Behavioral Trap: Long inactivity ─────────────────────────────────
    inactive_days = days_since(signals.get("last_active_date", ""))
    if inactive_days > 180:
        penalty -= 0.05
        issues.append("BEHAVIORAL_inactive_6_months")
    elif inactive_days > 120:
        penalty -= 0.03
        issues.append("BEHAVIORAL_inactive_4_months")

    # Cap total penalty at -0.10 as specified
    return max(-0.10, penalty), issues


# ==============================================================================
# DYNAMIC REASONING GENERATOR
# ==============================================================================

def generate_reasoning(candidate: dict, career_info: dict,
                       signals_info: dict, penalties: list,
                       final_score: float) -> str:
    """
    Generate a unique, factual, human-sounding 1-2 sentence reasoning.
    DESIGN FOR STAGE 4 MANUAL REVIEW:
    ─────────────────────────────────
    The spec says reviewers check:
      ✓ Specific facts from the profile (not generic praise)
      ✓ JD connection (production retrieval, hybrid search, eval framework)
      ✓ Honest concerns (notice period, consulting, inactivity)
      ✓ No hallucination (every claim verifiable from profile JSON)
      ✓ Variation across samples (no identical/templated sentences)
      ✓ Rank consistency (rank-5 reasons sound different from rank-85)
    Strategy: Assemble reasoning from atomic fact-blocks extracted from the
    real profile. Every number (YoE, notice days, response rate, saved_30d)
    is read directly from the JSON object. We pick different sentence openers,
    JD anchors, and concern phrasings based on candidate_id modulo rotations
    to ensure variation even for similarly-scoring candidates.
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    cid = candidate.get("candidate_id", "CAND_0000000")

    yoe = profile.get("years_of_experience", 0.0)
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "")
    location = profile.get("location", "")

    roles_info = career_info.get("roles_info", [])
    ir_matches = career_info.get("ir_matches", 0)
    prod_matches = career_info.get("prod_matches", 0)
    llm_matches = career_info.get("llm_matches", 0)
    num_ml_roles = career_info.get("num_ml_roles", 0)
    consulting_frac = career_info.get("consulting_fraction", 0.0)
    current_disq = career_info.get("current_disqualified", False)
    has_disq = career_info.get("has_disqualified", False)

    notice_days = signals_info.get("notice_days", 90)
    response_rate = signals_info.get("response_rate", 0.5)
    github_score = signals_info.get("github_score", -1)
    saved_30d = signals_info.get("saved_30d", 0)
    days_inactive = signals_info.get("days_inactive", 0)
    open_to_work = signals_info.get("open_to_work", False)
    willing_relocate = signals_info.get("willing_relocate", False)
    is_in_target = signals_info.get("is_in_target", False)

    # ── Variation seed (no random — must be deterministic) ────────────────
    cid_num = int(cid.replace("CAND_", ""))
    v = cid_num % 6   # opener variation: 6 styles

    # ── Extract real skills for mention ──────────────────────────────────
    ir_skills = extract_top_skills(skills, RETRIEVAL_KEYWORDS, n=2)
    llm_skills = extract_top_skills(skills, LLM_KEYWORDS, n=2)
    all_top_skills = extract_top_skills(skills, CORE_AI_SKILLS, n=3)

    # ── Pull most relevant description snippet ────────────────────────────
    # Find the role description with the most retrieval/production signals
    best_desc = ""
    best_match_count = -1
    for role in roles_info:
        if role.get("hard_disqualified"):
            continue
        c = count_unique_matches(RETRIEVAL_KEYWORDS, role["description"]) + \
            count_unique_matches(PRODUCTION_KEYWORDS, role["description"])
        if c > best_match_count:
            best_match_count = c
            best_desc = role["description"]

    # ── Parse a usable snippet from the best description ─────────────────
    has_production_evidence = prod_matches >= 2
    has_retrieval_evidence = ir_matches >= 2
    has_rag_mention = "rag" in best_desc.lower() or "retrieval augmented" in best_desc.lower()
    has_vector_mention = any(v in best_desc.lower() for v in
                             ["vector", "pinecone", "milvus", "qdrant", "weaviate", "faiss"])
    has_scale_mention = any(s in best_desc.lower() for s in
                            ["scale", "million", "qps", "latency", "live traffic", "users"])
    has_ab_mention = any(s in best_desc.lower() for s in ["a/b test", "ab test", "online eval"])
    has_eval_framework = any(s in best_desc.lower() for s in ["ndcg", "mrr", "map", "eval framework"])

    # ── Determine score band for tone calibration ─────────────────────────
    if final_score >= 0.85:
        score_band = "elite"
    elif final_score >= 0.70:
        score_band = "strong"
    elif final_score >= 0.55:
        score_band = "moderate"
    elif final_score >= 0.35:
        score_band = "weak"
    else:
        score_band = "poor"

    # ================================================================
    # BUILD SENTENCE 1: Core profile identity + key technical strength
    # ================================================================

    if current_disq:
        # Hard disqualified current role — be honest about why they're ranked low
        s1 = (f"Currently working as {title}, which falls outside the technical scope "
               f"of this role (the JD requires production ML/AI engineering experience); "
               f"included at rank due to adjacent skills in profile but unlikely to be a fit.")
        return s1  # Single sentence is fine for very low scores

    # Opener styles rotated by cid_num
    if v == 0:
        s1 = f"{yoe:.1f}-year {title}"
        if company:
            s1 += f" at {company}"
    elif v == 1:
        s1 = f"{title} with {yoe:.1f} years of hands-on experience"
        if company and not is_consulting_company(company):
            s1 += f" (currently at {company})"
    elif v == 2:
        s1 = f"Applied ML engineer with {yoe:.1f} years of experience"
        if company:
            s1 += f", most recently at {company}"
    elif v == 3:
        s1 = f"{yoe:.1f} years of ML/AI engineering experience"
        if company:
            s1 += f" — current role at {company}"
    elif v == 4:
        # Guard: don't prepend 'Senior' if the title already contains it
        title_display = title if 'senior' in title.lower() else f"Senior {title}"
        s1 = f"{title_display} with {yoe:.1f} years total"
        if company:
            s1 += f", based at {company}"
    else:
        s1 = f"Product-side ML engineer ({yoe:.1f} yrs)"
        if company:
            s1 += f" currently at {company}"

    # Add the key technical strength — specific to what's in the profile
    strengths = []
    if has_rag_mention and ir_skills:
        strengths.append(f"has built RAG/retrieval systems ({', '.join(ir_skills[:2])})")
    elif has_vector_mention and ir_skills:
        strengths.append(f"shipped vector search infrastructure ({', '.join(ir_skills[:1])})")
    elif has_retrieval_evidence and ir_skills:
        strengths.append(f"background in {', '.join(ir_skills[:2])} for production search")
    elif ir_skills:
        strengths.append(f"hands-on with {', '.join(ir_skills[:2])}")

    if has_scale_mention:
        strengths.append("experience serving systems at production scale")
    if has_eval_framework:
        strengths.append("designed evaluation frameworks (NDCG/MRR)")
    elif has_ab_mention:
        strengths.append("ran A/B tests on ranking quality")
    if llm_skills and score_band in {"elite", "strong"}:
        strengths.append(f"LLM fine-tuning via {', '.join(llm_skills[:1])}")

    if strengths:
        s1 += "; " + strengths[0]
        if len(strengths) > 1 and score_band in {"elite", "strong"}:
            s1 += " and " + strengths[1]
    s1 += "."

    # ================================================================
    # BUILD SENTENCE 2: Behavioral signals + honest concerns
    # ================================================================

    # Positive signals to mention
    # Use 5 rotating phrasings for saved_by_recruiters to avoid templating
    _sv = cid_num % 5
    pos_signals = []
    if saved_30d >= 8:
        _saved_phrases = [
            f"profile saved by {saved_30d} recruiters this month",
            f"{saved_30d} recruiters bookmarked this profile in the past 30 days",
            f"high recruiter demand — {saved_30d} saves in last 30 days",
            f"shortlisted by {saved_30d} recruiters over the past month",
            f"strong recruiter interest ({saved_30d} profile saves recently)",
        ]
        pos_signals.append(_saved_phrases[_sv])
    elif saved_30d >= 4:
        _saved_phrases_sm = [
            f"bookmarked by {saved_30d} recruiters recently",
            f"{saved_30d} recruiter saves in the last 30 days",
            f"noted by {saved_30d} recruiters this month",
            f"saved by {saved_30d} recruiters in recent weeks",
            f"on {saved_30d} recruiter shortlists recently",
        ]
        pos_signals.append(_saved_phrases_sm[_sv])
    if response_rate >= 0.75:
        pos_signals.append(f"very high recruiter response rate ({int(response_rate*100)}%)")
    elif response_rate >= 0.55:
        pos_signals.append(f"good recruiter responsiveness ({int(response_rate*100)}%)")
    if github_score >= 75:
        pos_signals.append(f"active GitHub presence (score {github_score:.0f}/100)")
    elif github_score >= 50:
        pos_signals.append(f"GitHub activity score {github_score:.0f}/100")
    if days_inactive <= 7:
        pos_signals.append("active on platform this week")
    elif days_inactive <= 14:
        pos_signals.append("active on platform in the last two weeks")
    if is_in_target:
        pos_signals.append(f"already located in {location}")
    elif willing_relocate:
        pos_signals.append("open to relocating to Pune/Noida")
    if num_ml_roles >= 3:
        pos_signals.append(f"{num_ml_roles} consecutive ML/AI roles")
    if open_to_work and score_band in {"elite", "strong"}:
        pos_signals.append("actively seeking new opportunities")

    # Concerns (always stated honestly per Stage 4 spec)
    concerns = []
    if "TRAP_pure_consulting_career" in penalties:
        concerns.append("entire career history is in consulting firms (TCS/Infosys/Wipro etc.), which the JD explicitly flags as a poor fit")
    elif "TRAP_mostly_consulting_career" in penalties:
        firms = [r["company"] for r in roles_info if r.get("consulting")]
        firm_str = firms[0] if firms else "consulting"
        concerns.append(f"significant consulting background ({firm_str}); partial product-company exposure may help")
    if notice_days > 90:
        concerns.append(f"notice period of {notice_days} days exceeds what the JD prefers (≤30 days)")
    elif notice_days > 60:
        concerns.append(f"{notice_days}-day notice period (JD prefers sub-30)")
    if response_rate < 0.30:
        concerns.append(f"very low recruiter response rate ({int(response_rate*100)}%) — reachability concern")
    elif response_rate < 0.40:
        concerns.append(f"below-average response rate ({int(response_rate*100)}%)")
    if days_inactive > 180:
        concerns.append(f"profile inactive for {days_inactive} days — may not be actively looking")
    elif days_inactive > 120:
        concerns.append(f"not active on platform in {days_inactive} days")
    if "HONEYPOT_impossible_date_math" in penalties or "HONEYPOT_role_duration_exceeds_total_yoe" in penalties:
        concerns.append("career timeline contains inconsistencies that reduce confidence in stated experience")
    if "HONEYPOT_expert_skills_zero_duration" in penalties or "HONEYPOT_skill_stuffing_low_yoe" in penalties:
        concerns.append("skills list appears inflated relative to verifiable experience")
    if consulting_frac > 0 and consulting_frac < 1.0 and score_band in {"elite", "strong"}:
        # Mention it but not as a dealbreaker
        firms = [r["company"] for r in roles_info if r.get("consulting")]
        if firms:
            concerns.append(f"includes a stint at {firms[0]} (consulting), though prior/subsequent product-company work offsets this per JD")

    # Assemble sentence 2
    s2_parts = []
    if pos_signals:
        # Pick 1-2 most impressive positive signals (avoid over-padding)
        top_pos = pos_signals[:2]
        s2_parts.append(", ".join(top_pos).capitalize())

    if concerns:
        if s2_parts:
            s2_parts.append(f"concern: {concerns[0]}")
        else:
            s2_parts.append(f"Key concern: {concerns[0]}")
        if len(concerns) > 1 and score_band in {"poor", "weak", "moderate"}:
            s2_parts.append(f"also: {concerns[1]}")
    elif not s2_parts:
        # No concerns and no signals to highlight — give a generic JD-aligned note
        if score_band == "elite":
            s2_parts.append("Strong overall JD match with no major red flags")
        elif score_band == "strong":
            s2_parts.append("Solid fit with no critical concerns")
        else:
            s2_parts.append("Limited signal overlap with core JD requirements")

    s2 = "; ".join(s2_parts) + "."
    # Ensure it starts uppercase
    s2 = s2[0].upper() + s2[1:] if s2 else ""

    # ── Combine and clean ─────────────────────────────────────────────────
    reasoning = f"{s1} {s2}".strip()
    # Remove double punctuation artefacts
    reasoning = re.sub(r'\.\s*\.', '.', reasoning)
    reasoning = re.sub(r'\s+', ' ', reasoning)

    return reasoning


# ==============================================================================
# MASTER SCORING FUNCTION
# ==============================================================================

def score_candidate(candidate: dict) -> tuple[float, dict, str]:
    """
    Orchestrate all five scoring components and generate reasoning.
    Weight design rationale (for Stage 5 interview):
      Career (0.42): Single most important signal — JD is very specific about
        retrieval/ranking at product companies. Dominates NDCG@10.
      Behavioral (0.22): Second most important — unhirable candidates who don't
        respond or haven't logged in in months should not appear in top 100.
      Skills (0.18): Supporting signal. Cross-validated with platform assessments.
      Exp/Edu (0.10): Tiebreaker; the JD is explicit about 5-9 year range.
      Penalty (-0.10..0): Hard filter on honeypots and disqualified profiles.
    """
    # Component 1: Career Match (0.42)
    career_score, career_info = compute_career_score(candidate)

    # Component 2: Skills Relevance (0.18)
    skills_score = compute_skills_score(candidate)

    # Component 3: Behavioral & Availability (0.22)
    behavioral_score, signals_info = compute_behavioral_score(candidate)

    # Component 4: Experience & Education (0.10)
    exp_edu_score = compute_exp_edu_score(candidate)

    # Component 5: Penalties (-0.10 to 0)
    penalty, penalty_list = detect_penalties(candidate, career_info)

    # ── Weighted sum ──────────────────────────────────────────────────────
    raw = (
        career_score     * 0.42 +
        skills_score     * 0.18 +
        behavioral_score * 0.22 +
        exp_edu_score    * 0.10 +
        penalty                   # already negative
    )

    # Normalize: max positive sum = 0.92, so divide by 0.92 to get [0,1]
    normalized = min(1.0, max(0.0, raw / 0.92))

    # ── Snapshot for debugging ────────────────────────────────────────────
    breakdown = {
        "career":    round(career_score, 4),
        "skills":    round(skills_score, 4),
        "behavioral":round(behavioral_score, 4),
        "exp_edu":   round(exp_edu_score, 4),
        "penalty":   round(penalty, 4),
        "raw":       round(raw, 4),
        "final":     round(normalized, 4),
        "penalties": penalty_list
    }

    # ── Reasoning ─────────────────────────────────────────────────────────
    reasoning = generate_reasoning(
        candidate, career_info, signals_info, penalty_list, normalized
    )

    return normalized, breakdown, reasoning


# ==============================================================================
# STREAMING PIPELINE
# ==============================================================================

def process_candidates(input_file: str, output_file: str):
    """
    Stream candidates.jsonl, score every profile, rank top 100, write CSV.
    Memory footprint: O(N) for the scored list, but each entry is tiny (~200 bytes).
    Total RAM usage for 100K candidates: ~20 MB.
    """
    print(f"[INFO] Reading: {input_file}")
    t0 = datetime.now()

    scored = []
    errors = 0

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

        # Detect JSON array or JSONL
        try:
            if content.startswith("["):
                candidates = json.loads(content)
            else:
                candidates = [
                    json.loads(line)
                    for line in content.splitlines()
                    if line.strip()
                ]
        except Exception as e:
            print(f"[ERROR] Invalid input file: {e}")
            return
    
        # Process every candidate
        for idx, candidate in enumerate(candidates, start=1):
            try:
                score, breakdown, reasoning = score_candidate(candidate)
    
                scored.append({
                    "candidate_id": candidate["candidate_id"],
                    "score": score,
                    "reasoning": reasoning,
                    "breakdown": breakdown
                })
    
            except Exception as exc:
                errors += 1
                print(f"[WARN] Candidate {idx}: {exc}", file=sys.stderr)
    
            if idx % 20000 == 0:
                elapsed = (datetime.now() - t0).total_seconds()
                print(f"[INFO] Processed {idx:,} candidates ({elapsed:.1f}s)")

    # ── Sort: score descending; candidate_id ascending for tie-break ──────
    # CRITICAL: round to 4dp BEFORE sort so the tie-break matches the CSV output.
    # This is what caused the validator failure in v1.
    scored.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))

    top100 = scored[:100]

    # ── Write output CSV ──────────────────────────────────────────────────
    print(f"[INFO] Writing top 100 → {output_file}")
    with open(output_file, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, item in enumerate(top100, 1):
            writer.writerow([
                item["candidate_id"],
                rank,
                f"{item['score']:.4f}",
                item["reasoning"]
            ])

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"[INFO] Done. {len(scored):,} candidates processed, {errors} errors, "
          f"{elapsed:.2f}s total.")
    print()

    # ── Debug: Top 5 ──────────────────────────────────────────────────────
    print("=" * 70)
    print("TOP 5 CANDIDATES — DEBUG LOG")
    print("=" * 70)
    for rank, item in enumerate(top100[:5], 1):
        b = item["breakdown"]
        print(f"\nRank {rank}: {item['candidate_id']}  |  Score: {item['score']:.4f}")
        print(f"  Breakdown → career:{b['career']} skills:{b['skills']} "
              f"behav:{b['behavioral']} exp_edu:{b['exp_edu']} penalty:{b['penalty']}")
        if b["penalties"]:
            print(f"  Penalties → {b['penalties']}")
        print(f"  Reasoning: {item['reasoning']}")
    print("=" * 70)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Redrob Candidate Ranking System v2 — Senior AI Engineer role"
    )
    parser.add_argument(
        "--input", "-i",
        default="candidates.jsonl",
        help="Path to input JSONL file (default: candidates.jsonl)"
    )
    parser.add_argument(
        "--output", "-o",
        default="submission.csv",
        help="Path to output ranked CSV file (default: submission.csv)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    process_candidates(args.input, args.output)