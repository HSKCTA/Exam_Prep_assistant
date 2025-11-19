from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from urllib.parse import urljoin
from rasa_sdk.types import DomainDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .sppu_scrapper import get_papers_from_sppu_tables 
from typing import List, Tuple
import os
import requests
from bs4 import BeautifulSoup
import tempfile
import re
import spacy
import PyPDF2
import random
import logging
from docx import Document
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import difflib


_TRANSFORMER_MODEL = None
def get_transformer_model():
    global _TRANSFORMER_MODEL
    if _TRANSFORMER_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _TRANSFORMER_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logging.getLogger("actions").warning(f"SentenceTransformer unavailable: {e}")
            _TRANSFORMER_MODEL = None
    return _TRANSFORMER_MODEL




nlp = spacy.load("en_core_web_sm")
logger = logging.getLogger(__name__)

import pdfplumber


def extract_clean_text(pdf_path):
    extracted_text = []
    watermark_candidates = Counter()

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
               
                for line in text.split("\n"):
                    watermark_candidates[line] += 1

    watermark_threshold = len(extracted_text) * 0.7  # Appears on 70%+ pages
    watermarks = {line for line, count in watermark_candidates.items() if count >= watermark_threshold}

    clean_text = "\n".join([line for text in extracted_text for line in text.split("\n") if line not in watermarks])

    return clean_text


def send_image_inline_or_fallback(dispatcher, image_path):
    """
    Try to read PNG and send as data URL (many web widgets render it).
    Fallback to sending file:// absolute path if data URL not supported.
    """
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"
        # Many channels will render this inline.
        dispatcher.utter_message(image=data_url)
        return True
    except Exception as e:
        # fallback: send file path (works in Rasa shell / some connectors)
        try:
            dispatcher.utter_message(image=f"file://{os.path.abspath(image_path)}")
            return True
        except Exception:
            dispatcher.utter_message(text="(Image generation succeeded but couldn't send image in this channel.)")
            return False


def extract_questions_and_marks(clean_text):
    questions = []
    
    # Main pattern: capture main question number and its associated block.
    main_pattern = re.compile(r"Q(\d+)\)\s*(.*?)(?=\n?Q\d+\)|\Z)", re.DOTALL | re.IGNORECASE)
    # Sub-question pattern: captures sub-question letter, the question text (across newlines), and marks.
    sub_pattern = re.compile(r"([a-z])\)\s*(.*?)\s*\[(\d+)\]", re.IGNORECASE | re.DOTALL)
    
    for main_match in main_pattern.finditer(clean_text):
        main_question_no = int(main_match.group(1))
        block_text = main_match.group(2)
        # Remove any "OR" separators from the block.
        block_text = re.sub(r"\bOR\b", "", block_text, flags=re.IGNORECASE)
        sub_matches = sub_pattern.findall(block_text)
        for sub_match in sub_matches:
            sub_question, question_text, marks = sub_match
            questions.append({
                "question_no": main_question_no,
                "sub_question": sub_question,
                "question": question_text.strip(),
                "marks": int(marks)
            })
    
    return questions

def clean_question_text(text):
    """Cleans up question text for analysis."""
    cleaned = re.sub(r"(?<=\s)\d{1,2}(?=\s)", " ", text)
    cleaned = re.sub(r"^\d{1,2}\s+", "", cleaned)
    cleaned = re.sub(r"\s+\d{1,2}$", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()

def format_questions(questions):
    # Group questions by main question number.
    grouped = {}
    for q in questions:
        grouped.setdefault(q["question_no"], []).append(q)
    
    formatted_output = ""
    # Iterate in order of question number.
    for q_no in sorted(grouped.keys()):
        formatted_output += f"Q{q_no}) "
        # Sort sub-questions by their letter.
        sub_questions = sorted(grouped[q_no], key=lambda x: x["sub_question"])
        first = True
        for sub in sub_questions:
            clean_text_val = clean_question_text(sub["question"])
            if first:
                formatted_output += f"{sub['sub_question']}) {clean_text_val} [{sub['marks']}]\n"
                first = False
            else:
                formatted_output += f"   {sub['sub_question']}) {clean_text_val} [{sub['marks']}]\n"
        formatted_output += "\n"
    return formatted_output

def process_multiple_papers(pdf_paths):
    """
    Extract structured question dicts from each paper, but return ONLY a list of question text strings
    for downstream NLP/clustering.
    """
    all_questions = []

    for pdf_path in pdf_paths:
        clean_text = extract_clean_text(pdf_path)
        extracted = extract_questions_and_marks(clean_text)

        for item in extracted:
            if isinstance(item, str):
                # rare case
                q_text = item.strip()
                if q_text:
                    all_questions.append(q_text)

            elif isinstance(item, dict):
                # Prefer the "question" key (your extractor's output)
                q_text = item.get("question")
                if isinstance(q_text, str) and q_text.strip():
                    all_questions.append(q_text.strip())

            # Ignore anything else (None, numbers, invalid objects)
            else:
                continue

    return all_questions



def cluster_similar_questions(questions: List[str], similarity_threshold=0.8):
    """
    Clusters questions using cosine similarity + AgglomerativeClustering.
    Compatible with older sklearn versions (uses affinity='precomputed').
    """

    # TF-IDF representation
    vect = TfidfVectorizer(stop_words='english')
    X = vect.fit_transform(questions)

    # cosine similarity matrix
    sim = cosine_similarity(X)

    # convert similarity → distance
    dist = 1 - sim

    # sklearn < 1.2 requires affinity='precomputed'
    clustering = AgglomerativeClustering(
        affinity='precomputed',
        linkage='average',
        distance_threshold=1 - similarity_threshold,
        n_clusters=None
    )

    labels = clustering.fit_predict(dist)
    return labels

def plot_most_frequent_clusters(questions, labels, top_n: int = 10, output_path: str = "cluster_analysis_plot.png"):
    """
    Robust plotting for question clusters.
    Accepts questions (iterable of str) and labels (iterable of ints or numpy array).
    Returns absolute path to saved image.
    """

    logger = logging.getLogger("actions.plot_most_frequent_clusters")

    # Coerce to lists to avoid numpy truthiness issues
    try:
        questions_list = list(questions)
    except Exception:
        raise ValueError("questions must be iterable")

    try:
        labels_list = list(labels)
    except Exception:
        raise ValueError("labels must be iterable")

    # explicit emptiness checks
    if len(questions_list) == 0 or len(labels_list) == 0:
        raise ValueError("questions and labels must be non-empty lists")

    if len(questions_list) != len(labels_list):
        raise ValueError("questions and labels must have the same length")

    # Count frequency per cluster and collect first-occurrence representative
    cluster_counter = Counter(labels_list)
    top_clusters = cluster_counter.most_common(top_n)

    reps = {}
    cluster_to_indices = defaultdict(list)
    for idx, lab in enumerate(labels_list):
        cluster_to_indices[lab].append(idx)
        if lab not in reps:
            reps[lab] = questions_list[idx]

    clusters = [f"Cluster {lab}" for lab, _ in top_clusters]
    freqs = [count for _, count in top_clusters]
    labels_for_text = [lab for lab, _ in top_clusters]
    rep_texts = [reps.get(lab, "") for lab in labels_for_text]

    labels = list(labels) 
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(clusters, freqs)
    plt.xlabel("Cluster")
    plt.ylabel("Frequency")
    plt.title("Top Frequently Asked Question Clusters")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # annotate bars with counts and short representative
    for bar, rep, freq in zip(bars, rep_texts, freqs):
        h = bar.get_height()
        rep_short = rep if len(rep) <= 120 else rep[:117] + "..."
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.5, f"{freq}\n{rep_short}", ha='center', va='bottom', fontsize=8)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    abs_path = os.path.abspath(output_path)
    logger.debug(f"Saved cluster plot to {abs_path}")
    return abs_path


import base64
import io

def clean_text_for_display(text: str) -> str:
    """
    Clean common OCR / extraction artifacts so question text is readable.
    - remove isolated digits/noise
    - remove stray control chars
    - collapse repeated punctuation/whitespace
    - preserve punctuation inside proper words
    """
    if not text:
        return ""

    s = str(text)

    # replace weird control characters/newlines with single space
    s = re.sub(r'[\r\n\t]+', ' ', s)

    # remove sequences like "0 0", multiple isolated digits, and digits inserted inside words
    # Strategy: remove standalone digit tokens and sequences of digits separated by spaces
    s = re.sub(r'(?:\b\d+\b[\s]*){1,}', ' ', s)

    # remove digits embedded between letters (common OCR corruption), e.g. "nor3mal" -> "normal"
    s = re.sub(r'(?<=\w)\d+(?=\w)', '', s)

    # collapse runs of non-alphanumeric punctuation except sentence punctuation
    s = re.sub(r'[^\w\s\.,\?\-:;()]+', ' ', s)

    # collapse multi punctuation e.g. "???" -> "?"
    s = re.sub(r'([?.!,;:-])\1+', r'\1', s)

    # normalize whitespace
    s = re.sub(r'\s{2,}', ' ', s).strip()

    # remove leading/trailing punctuation leftover
    s = s.strip(" -:.;,")

    # Capitalize first letter (nice formatting)
    if s:
        s = s[0].upper() + s[1:]

    return s

def summarize_clusters_pretty(questions, labels, top_n: int = 6, max_display_len: int = 400):
    """
    Returns (summary_list, markdown_text)
    summary_list: [(rep_text, count, sample_text), ...]
    markdown_text: readable numbered list (str)
    """
    questions_list = list(questions)
    labels_list = list(labels)

    # safety
    if len(questions_list) == 0 or len(labels_list) == 0:
        return [], "No questions found."

    n = min(len(questions_list), len(labels_list))
    questions_list = questions_list[:n]
    labels_list = labels_list[:n]

    cluster_counter = Counter(labels_list)
    top = cluster_counter.most_common(top_n)

    cluster_to_indices = defaultdict(list)
    for idx, lab in enumerate(labels_list):
        cluster_to_indices[lab].append(idx)

    summary = []
    md_lines = []
    md_lines.append("**Top recurring question types:**\n")
    for rank, (lab, count) in enumerate(top, start=1):
        idxs = cluster_to_indices.get(lab, [])
        rep_text_raw = questions_list[idxs[0]] if idxs else ""
        sample_raw = questions_list[idxs[0]] if idxs else ""
        rep_text = clean_text_for_display(rep_text_raw)
        sample_text = clean_text_for_display(sample_raw)

        # ensure we don't cut important words — limit to max_display_len
        if len(rep_text) > max_display_len:
            rep_short = rep_text[:max_display_len-3].rsplit(' ',1)[0] + "..."
        else:
            rep_short = rep_text

        summary.append((rep_text, count, sample_text))

        # build markdown: numbered entry with count and representative question
        md_lines.append(f"{rank}. ({count} times) {rep_short}\n")

    md_text = "\n".join(md_lines)
    return summary, md_text



class ActionFetchAndAnalyze(Action):
    def name(self) -> Text:
        return "action_fetch_and_analyze"

    # keep your existing _find_link_on_page() if you want, but we will not use it here
    # ...

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        MAX_DOWNLOAD_PAPERS = 10  
        # 1. get slots
        dept_slot = tracker.get_slot("department")
        sem_slot = tracker.get_slot("semester")
        subject_slot = tracker.get_slot("selected_subject")
        pattern_slot = tracker.get_slot("pattern")

        if not all([dept_slot, sem_slot, subject_slot, pattern_slot]):
            dispatcher.utter_message(text="I'm missing some information. I need department, semester, subject, and pattern.")
            return []

        dispatcher.utter_message(text=f"Searching for '{subject_slot}' papers ({pattern_slot}) in {dept_slot} — sem {sem_slot}...")

        # 2. scrape
        try:
            pdf_urls, matched_headers, main_page, table_pages = get_papers_from_sppu_tables(
                dept_slot, sem_slot, subject_slot, pattern_slot
            )
        except Exception as e:
            logger.exception("Scraper error")
            dispatcher.utter_message(text="Sorry, I ran into an error while searching the papers.")
            return []

        if not pdf_urls:
            dispatcher.utter_message(text=f"Couldn't find any papers for {subject_slot} ({pattern_slot}). I searched: {main_page}")
            return []

        dispatcher.utter_message(text=f"Found {len(pdf_urls)} paper links. Downloading up to {MAX_DOWNLOAD_PAPERS} files for analysis...")

        # 3. download (may raise)
        tmp_pdf_paths: List[str] = []
        image_path = "cluster_analysis_plot.png"
        try:
            tmp_pdf_paths = download_pdfs(pdf_urls, max_papers=MAX_DOWNLOAD_PAPERS)
        except Exception as e:
            logger.exception("Error while downloading PDFs")
            dispatcher.utter_message(text="I couldn't download the papers due to a network error.")
            return []

        if not tmp_pdf_paths:
            dispatcher.utter_message(text="I couldn't download any of the found papers.")
            return []

        logger.debug(f"Downloaded {len(tmp_pdf_paths)} files: {tmp_pdf_paths}")
        dispatcher.utter_message(text=f"Downloaded {len(tmp_pdf_paths)} papers. Starting analysis...")

                # --- 4. analysis (inside try so finally cleans up) ---
        try:
            # 4.1 extract questions
            try:
                all_questions = process_multiple_papers(tmp_pdf_paths)
            except Exception:
                logger.exception("process_multiple_papers failed")
                dispatcher.utter_message(text="I failed to extract questions from the downloaded papers.")
                return []

            if not all_questions:
                dispatcher.utter_message(text="I analyzed the papers but couldn't extract questions reliably.")
                return []

            logger.debug(f"Extracted {len(all_questions)} questions. Starting clustering.")

            # 4.2 clustering (with internal fallback handled inside function if needed)
            try:
                labels = cluster_similar_questions(all_questions, similarity_threshold=0.8)
                labels = list(labels)  # ensure it's a plain Python list (avoid numpy truthiness problems)
            except Exception:
                logger.exception("Clustering failed; falling back to TF-IDF clustering")
                # If you have a fallback function, call it here; for now try the TF-IDF fallback
                labels = list(_fallback_cluster_labels(all_questions))  # ensure you have this available

            # 4.3 plotting (if it fails, continue without image)
            image_path = "cluster_analysis_plot.png"
            try:
                plot_most_frequent_clusters(all_questions, labels, top_n=10, output_path=image_path)
            except Exception:
                logger.exception("Plot generation failed (continuing without image)")
                image_path = None

            # 4.4 produce pretty summary + send image inline/fallback
            try:
                summary_list, markdown_summary = summarize_clusters_pretty(all_questions, labels, top_n=6)

                # top-line confirmation
                dispatcher.utter_message(
                    text=f"Analysis Complete! I found and clustered {len(all_questions)} questions from {len(tmp_pdf_paths)} papers."
                )

                # send nicely formatted summary (markdown)
                dispatcher.utter_message(text=markdown_summary)

                # attempt to send inline image (data URL) and fallback to file path
                if image_path and os.path.exists(image_path):
                    ok = send_image_inline_or_fallback(dispatcher, image_path)
                    if not ok:
                        dispatcher.utter_message(text="(Image generated but couldn't be sent inline in this channel.)")
                else:
                    dispatcher.utter_message(text="(No cluster plot available.)")

            except Exception:
                logger.exception("Failed to create/send pretty summary or image")
                # graceful fallback: minimal text + path image if present
                try:
                    msg = f"Found {len(all_questions)} questions clustered into {len(set(labels))} groups."
                    dispatcher.utter_message(text=msg)
                    if image_path and os.path.exists(image_path):
                        dispatcher.utter_message(image=f"file://{os.path.abspath(image_path)}")
                except Exception:
                    dispatcher.utter_message(text="Analysis completed, but I couldn't generate the summary or image for display.")

        except Exception:
            logger.exception("Error during analysis")
            dispatcher.utter_message(text="Sorry — an error occurred during analysis.")
        finally:
            # cleanup downloaded PDF temp files only (keep plot image for viewing)
            for p in tmp_pdf_paths:
                try:
                    os.remove(p)
                except OSError:
                    logger.debug(f"Failed to remove temp file {p}")
            # DO NOT delete the image file here if you want to view it after the action.
            # If you later want to remove it, do so in a separate cleanup routine or on-demand.


        # Clear slots
        return [SlotSet(s, None) for s in ["department", "semester", "selected_subject", "pattern"]]



class Config:
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = ['.pdf', '.docx']
    ANALYSIS_TIMEOUT = 300  # 5 minutes

class ActionHandleNavigation(Action):
    def name(self) -> Text:
        return "action_handle_navigation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        page_mapping = {
            "study materials": "https://yourwebsite.com/materials",
            "practice tests": "https://yourwebsite.com/tests",
            "analysis reports": "https://yourwebsite.com/reports",
            "account settings": "https://yourwebsite.com/account"
        }
        
        requested_page = next(tracker.get_latest_entity_values("page"), "").lower()
        
        if requested_page in page_mapping:
            dispatcher.utter_message(
                text=f"Redirecting to {requested_page.replace('_', ' ').title()}...",
                buttons=[{"title": "Click Here", "url": page_mapping[requested_page]}]
            )
        else:
            dispatcher.utter_message(
                text="Here are our main sections:",
                buttons=[{"title": k.title(), "url": v} for k,v in page_mapping.items()]
            )
        
        return [SlotSet("current_page", requested_page)]


class ActionGenerateMockTest(Action):
    def name(self) -> Text:
        return "action_generate_mock_test"

    def run(self, dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            # Get slot values with null checks
            subject = (tracker.get_slot("selected_subject") or "general").lower()
            question_count = int(tracker.get_slot("question_count") or 10)
            difficulty = (tracker.get_slot("difficulty_level") or "medium").lower()
            time_limit = tracker.get_slot("time_limit") or "30"  # Default to 30 minutes

            # Validate inputs
            if not subject or subject not in ["math", "physics", "chemistry"]:
                raise ValueError(f"Invalid subject: {subject}")
                
            if question_count < 1 or question_count > 50:
                raise ValueError(f"Invalid question count: {question_count}")

            # Generate test content
            test_content = self.generate_test_content(
                subject=subject,
                num_questions=question_count,
                difficulty=difficulty
            )

            # Format response
            response = (
                f"📝 Mock Test for {subject.capitalize()} ({difficulty.capitalize()})\n"
                f"⏰ Time Limit: {time_limit} minutes\n\n"
                f"{test_content}"
            )

            dispatcher.utter_message(text=response)
            
            # PDF generation option
            dispatcher.utter_message(
                text="Would you like a PDF version?",
                buttons=[
                    {"title": "Yes", "payload": "/affirm"},
                    {"title": "No", "payload": "/deny"}
                ]
            )

        except Exception as e:
            logger.error(f"Test generation failed: {str(e)}", exc_info=True)
            dispatcher.utter_message(
                text=f"Failed to generate test: {str(e)}. " +
                     "Please check your parameters and try again."
            )
        
        return []

    def generate_test_content(self, subject: str, num_questions: int, difficulty: str) -> str:
        """Generate mock test questions with validation"""
        question_bank = self.load_question_bank(subject, difficulty)
        
        if not question_bank:
            raise ValueError(f"No questions available for {subject} ({difficulty})")
            
        selected_questions = random.sample(
            question_bank, 
            min(num_questions, len(question_bank))
        )
        
        return self.format_questions(selected_questions)

    def format_questions(self, questions: List[Dict]) -> str:
        """Format questions with proper numbering"""
        content = ""
        for i, q in enumerate(questions, 1):
            content += f"{i}. {q['question']}\n"
            if q.get('options'):
                for opt_idx, option in enumerate(q['options'], 1):
                    content += f"   {chr(96 + opt_idx)}) {option}\n"
            content += "\n"
        return content


    def load_question_bank(self, subject: str, difficulty: str) -> List[Dict]:
        """Load sample questions from a database or file"""
        # Example question structure
        question_bank = {
            "math": {
                "easy": [
                    {
                        "question": "Solve for x: 2x + 5 = 15",
                        "type": "short",
                        "answer": "5"
                    },
                    {
                        "question": "What is the area of a square with side 4cm?",
                        "type": "mcq",
                        "options": ["16cm²", "20cm²", "8cm²", "12cm²"],
                        "answer": "16cm²"
                    }
                ],
                "medium": [
                    {
                        "question": "Find the derivative of f(x) = 3x² + 2x",
                        "type": "short",
                        "answer": "6x + 2"
                    }
                ]
            },
            "physics": {
                "easy": [
                    {
                        "question": "State Newton's first law of motion",
                        "type": "short",
                        "answer": "An object at rest stays at rest..."
                    }
                ]
            }
        }

        return question_bank.get(subject, {}).get(difficulty, [])
    



DOWNLOAD_TIMEOUT = 20
MAX_WORKERS = 6
MAX_PAPERS = 50  # safety cap

def _download_single(url: str, headers: dict) -> Tuple[str, str]:
    """
    Download a single URL into a temp file.
    Returns (temp_path, url) on success, or (None, url) on failure.
    """
    try:
        resp = requests.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()
        # optional: basic content-type check
        ctype = resp.headers.get("Content-Type", "").lower()
        if "pdf" not in ctype and not url.lower().endswith(".pdf"):
            # Not strictly fatal; sometimes servers don't set Content-Type correctly,
            # but we log/skip here to be conservative.
            return None, url

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    tf.write(chunk)
            tmp_path = tf.name
        return tmp_path, url
    except Exception as e:
        # log if you have logger
        # logger.warning(f"Failed to download {url}: {e}")
        return None, url

def download_pdfs(urls: List[str], max_papers: int = MAX_PAPERS, workers: int = MAX_WORKERS) -> List[str]:
    """
    Download up to max_papers from urls in parallel and return list of local file paths.
    Skips failures. Caller must delete returned files after use.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SPPUBot/1.0)"}
    urls = urls[:max_papers]
    paths = []
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_download_single, u, headers): u for u in urls}
        for fut in as_completed(futures):
            tmp_path, url = fut.result()
            if tmp_path:
                paths.append(tmp_path)
            else:
                # optionally log failures
                pass
    return paths

class ActionAnalyzeQuestionPaper(Action):
    def name(self) -> Text:
        return "action_analyze_question_paper"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        file_path = tracker.get_slot("uploaded_file")

        try:
            # --- 1. VALIDATION BLOCK ---
            if not file_path:
                raise ValueError("File path is missing.")
            if not os.path.exists(file_path):
                raise ValueError("File not found on the server.")

            # Optional: replace Config.* with concrete values if Config not defined
            try:
                max_size = Config.MAX_FILE_SIZE
                allowed_ext = Config.ALLOWED_EXTENSIONS
            except Exception:
                max_size = 50 * 1024 * 1024  # 50 MB fallback
                allowed_ext = [".pdf", ".docx"]

            if os.path.getsize(file_path) > max_size:
                raise ValueError(f"File exceeds {max_size//1024//1024}MB limit.")

            if not any(file_path.lower().endswith(ext) for ext in allowed_ext):
                raise ValueError("Unsupported file type. Please use PDF or DOCX.")

            # --- 2. ANALYSIS BLOCK ---
            text = self._extract_text(file_path)
            if not text or len(text.strip()) < 20:
                raise RuntimeError("Extracted text is too short or empty.")

            questions = self._process_text(text)
            if not questions:
                raise RuntimeError("No questions could be extracted from the file.")

            # cluster labels (uses transformer if available, else TF-IDF fallback)
            cluster_labels = self._get_cluster_labels(questions)
            image_path = self._plot_question_clusters(questions, cluster_labels, top_n=10)

            analysis = {
                "topics": self._identify_topics(questions),
                "frequent_questions": self._find_frequent_questions(questions),
                "difficulty": self._estimate_difficulty(questions),
                "question_types": self._categorize_question_types(questions),
                "cluster_plot": os.path.abspath(image_path)
            }

            dispatcher.utter_message(text=self._format_analysis(analysis))
            # use file:// absolute path for Rasa file uploads
            abs_image = os.path.abspath(image_path)
            dispatcher.utter_message(image=f"file://{abs_image}")

            return [SlotSet("analysis_results", analysis)]

        except Exception as e:
            logger.exception(f"File analysis failed: {str(e)}")
            dispatcher.utter_message(text=f"An error occurred: {str(e)}")
            return [SlotSet("uploaded_file", None)]

    def _extract_text(self, file_path: Text) -> Text:
        """Extract text from supported file types (robust to None)."""
        try:
            if file_path.lower().endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = []
                    for page in reader.pages:
                        txt = page.extract_text() or ""
                        pages.append(txt)
                    return " ".join(pages)
            elif file_path.lower().endswith('.docx'):
                doc = Document(file_path)
                return " ".join([para.text for para in doc.paragraphs if para.text])
            else:
                raise RuntimeError("Unsupported file type for extraction.")
        except Exception as e:
            raise RuntimeError(f"Text extraction failed: {str(e)}")

    def _process_text(self, text: Text) -> List[Text]:
        """Clean and split questions (heuristic)."""
        # remove weird characters but keep punctuation that helps segmentation
        cleaned = re.sub(r'[^\w\s\.\?\(\)]+', ' ', text)
        # split heuristically on common question markers
        parts = re.split(r'(?:Q\s*\d+[:.\)]|\bQuestion\s*\d+[:.\)])', cleaned, flags=re.IGNORECASE)
        # filter and strip
        questions = [p.strip() for p in parts if p and len(p.strip()) > 20]
        return questions

    def _identify_topics(self, questions: List[Text]) -> List[Text]:
        """LDA Topic Modeling; guard small inputs."""
        if len(questions) < 3:
            # not enough data for LDA; return top tokens from TF-IDF
            vect = TfidfVectorizer(max_df=0.95, min_df=1, stop_words='english', max_features=50)
            dtm = vect.fit_transform(questions)
            terms = vect.get_feature_names_out()
            top_terms = terms[:5].tolist() if len(terms) else []
            return [", ".join(top_terms)]
        try:
            vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            dtm = vectorizer.fit_transform(questions)
            lda = LatentDirichletAllocation(n_components=5, random_state=42)
            lda.fit(dtm)
            topics = []
            for topic in lda.components_:
                top_idx = topic.argsort()[-3:]
                topics.append(", ".join([vectorizer.get_feature_names_out()[i] for i in top_idx]))
            return topics
        except Exception as e:
            logger.warning(f"LDA topic extraction failed: {e}")
            return []

    def _find_frequent_questions(self, questions: List[Text]) -> List[Text]:
        """Return representative questions per cluster (uses transformer if available)."""
        question_texts = [self._clean_question_text(q) for q in questions]
        if not question_texts:
            return []

        model = get_transformer_model()
        if model is None:
            labels = self._fallback_cluster_labels(question_texts)
            # derive representatives
            from collections import defaultdict
            clusters_map = defaultdict(list)
            for idx, lab in enumerate(labels):
                clusters_map[lab].append(idx)
            reps = [question_texts[idxs[0]] for idxs in clusters_map.values()]
            return reps

        try:
            embeddings = model.encode(question_texts, convert_to_tensor=False)
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(embeddings)
            threshold = 0.8
            n = len(question_texts)
            visited = [False] * n
            reps = []
            for i in range(n):
                if not visited[i]:
                    visited[i] = True
                    reps.append(question_texts[i])
                    for j in range(i+1, n):
                        if not visited[j] and sim[i][j] >= threshold:
                            visited[j] = True
            return reps
        except Exception as e:
            logger.warning(f"Transformer clustering failed, using TF-IDF fallback: {e}")
            labels = self._fallback_cluster_labels(question_texts)
            from collections import defaultdict
            clusters_map = defaultdict(list)
            for idx, lab in enumerate(labels):
                clusters_map[lab].append(idx)
            reps = [question_texts[idxs[0]] for idxs in clusters_map.values()]
            return reps

    def _estimate_difficulty(self, questions: List[Text]) -> Text:
        """Heuristic difficulty estimation"""
        if not questions:
            return "Unknown"
        avg_length = np.mean([len(q.split()) for q in questions])
        return "Advanced" if avg_length > 50 else "Intermediate" if avg_length > 25 else "Basic"

    def _categorize_question_types(self, questions: List[Text]) -> Dict[Text, int]:
        patterns = {
            'Definition': r'\bdefine\b|\bwhat is\b|\bexplain\b',
            'Calculation': r'\bcalculate\b|\bsolve\b|\bcompute\b|\bformula\b',
            'Problem Solving': r'\bprove\b|\bdemonstrate\b|\bsolve the problem\b',
            'Comparison': r'\bcompare\b|\bcontrast\b|\bdifference between\b',
            'Enumeration': r'\blist\b|\bname\b|\bgive examples\b'
        }
        return {q_type: sum(1 for q in questions if re.search(pattern, q, re.IGNORECASE))
                for q_type, pattern in patterns.items()}

    def _format_analysis(self, analysis: Dict) -> Text:
        return (
            f"📊 Analysis Report:\n\n"
            f"🔍 Top Topics:\n{chr(10).join(analysis.get('topics', []))}\n\n"
            f"📌 Frequent Questions:\n{chr(10).join(analysis.get('frequent_questions', []))}\n\n"
            f"📈 Difficulty: {analysis.get('difficulty','Unknown')}\n\n"
            f"🧩 Question Types:\n{chr(10).join(f'- {k}: {v}' for k,v in analysis.get('question_types',{}).items())}\n\n"
            f"🖼 Cluster Plot saved at: {analysis.get('cluster_plot')}"
        )

    def _clean_question_text(self, text: Text) -> Text:
        cleaned = re.sub(r"(?<=\s)\d{1,2}(?=\s)", " ", text)
        cleaned = re.sub(r"^\d{1,2}\s+", "", cleaned)
        cleaned = re.sub(r"\s+\d{1,2}$", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    def _get_cluster_labels(self, questions: List[Text]) -> List[int]:
        question_texts = [self._clean_question_text(q) for q in questions]
        if not question_texts:
            return []

        model = get_transformer_model()
        if model is None:
            return self._fallback_cluster_labels(question_texts)

        try:
            embeddings = model.encode(question_texts, convert_to_tensor=False)
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(embeddings)
            threshold = 0.8
            n = len(question_texts)
            visited = [False] * n
            labels = [-1] * n
            cluster_id = 0
            for i in range(n):
                if not visited[i]:
                    labels[i] = cluster_id
                    visited[i] = True
                    for j in range(i+1, n):
                        if not visited[j] and sim[i][j] >= threshold:
                            labels[j] = cluster_id
                            visited[j] = True
                    cluster_id += 1
            return labels
        except Exception as e:
            logger.warning(f"Transformer clustering failed: {e}. Falling back to TF-IDF.")
            return self._fallback_cluster_labels(question_texts)

    def _plot_question_clusters(self, questions: List[Text], cluster_labels: List[int], top_n=10) -> str:
        from collections import Counter
        if not cluster_labels:
            # nothing to plot
            image_path = "question_clusters.png"
            plt.figure(figsize=(6,2))
            plt.text(0.5,0.5,"No clusters found", ha='center')
            plt.axis('off')
            plt.savefig(image_path, bbox_inches='tight')
            plt.close()
            return image_path

        cluster_counter = Counter(cluster_labels)
        top_clusters = cluster_counter.most_common(top_n)
        clusters = [f"Cluster {label}" for label, _ in top_clusters]
        frequencies = [count for _, count in top_clusters]

        plt.figure(figsize=(12, 6))
        plt.bar(clusters, frequencies)
        plt.xlabel("Clusters")
        plt.ylabel("Frequency")
        plt.title("Top Frequently Asked Question Types")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        image_path = "question_clusters.png"
        plt.savefig(image_path)
        plt.close()
        return image_path

    def _fallback_cluster_labels(self, questions: List[Text], distance_threshold: float = 0.25) -> List[int]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity

        texts = [self._clean_question_text(q) for q in questions]
        if not texts:
            return []

        vect = TfidfVectorizer(ngram_range=(1,2), max_features=2000, stop_words='english')
        X = vect.fit_transform(texts)
        sim = cosine_similarity(X)
        dist = 1 - sim

        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity='precomputed',
            linkage='average',
            distance_threshold=distance_threshold
        )
        try:
            labels = clustering.fit_predict(dist)
            return labels.tolist()
        except Exception as e:
            logger.warning(f"Fallback clustering failed: {e}")
            return list(range(len(texts)))



    # Remaining action classes (StudyPlan, MockTest, etc.) with similar improvements
    # [Include all other action classes from previous version with enhanced error handling]

class ValidateMockTestForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_mock_test_form"

    def validate_question_count(self, slot_value: Any, dispatcher: CollectingDispatcher,
                               tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        try:
            count = int(slot_value)
            if 5 <= count <= 50:
                return {"question_count": count}
            dispatcher.utter_message(text="Please choose between 5-50 questions")
        except ValueError:
            dispatcher.utter_message(text="Please enter a valid number")
        return {"question_count": None}

    def validate_time_limit(self, slot_value: Any, dispatcher: CollectingDispatcher,
                           tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        try:
            minutes = int(slot_value)
            if 5 <= minutes <= 180:
                return {"time_limit": minutes}
            dispatcher.utter_message(text="Please choose between 5-180 minutes")
        except ValueError:
            dispatcher.utter_message(text="Please enter a valid number")
        return {"time_limit": None}

    def validate_difficulty_level(self, slot_value: Any, dispatcher: CollectingDispatcher,
                                 tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        if slot_value.lower() in ["easy", "medium", "hard"]:
            return {"difficulty_level": slot_value.lower()}
        dispatcher.utter_message(text="Please choose: easy/medium/hard")
        return {"difficulty_level": None}


class ActionStoreFeedback(Action):
    def name(self) -> Text:
        return "action_store_feedback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        feedback_type = next(tracker.get_latest_entity_values("feedback_type"), "general")
        feedback_text = tracker.latest_message.get('text')
        
        # Store feedback with type classification
        with open("feedback.txt", "a") as f:
            f.write(f"{datetime.now()} - {feedback_type} - {feedback_text}\n")
        
        return []

# [Include other form validation classes with similar enhancements]

class ValidateStudyPlanForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_study_plan_form"

    def validate_selected_subject(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> Dict[Text, Any]:
        """Validate subject slot."""
        valid_subjects = ["math", "physics", "chemistry", "biology"]
        if slot_value.lower() in valid_subjects:
            return {"subject": slot_value}
        else:
            dispatcher.utter_message(text="Please enter a valid subject (e.g., Math, Physics).")
            return {"subject": None}

    def validate_study_duration(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> Dict[Text, Any]:
        """Validate duration slot."""
        if slot_value.isdigit() and 1 <= int(slot_value) <= 12:
            return {"duration": slot_value}
        else:
            dispatcher.utter_message(text="Please enter a duration between 1 and 12 hours.")
            return {"duration": None}
        



class ValidatePaperAnalysisForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_paper_analysis_form"

    def validate_department(self, slot_value: Any, dispatcher: CollectingDispatcher,
                            tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if not slot_value:
            dispatcher.utter_message(text="Please tell me the department (for example: Computer Engineering, IT).")
            return {"department": None}
        val_norm = re.sub(r"\s+", " ", str(slot_value)).title()
        return {"department": val_norm}

    def validate_semester(self, slot_value: Any, dispatcher: CollectingDispatcher,
                          tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if not slot_value:
            dispatcher.utter_message(text="Which semester? (e.g., Sem 3)")
            return {"semester": None}

        raw = str(slot_value).strip().lower()

        # If the user explicitly writes a 4-digit year or mentions 'pattern' or 'endsem',
        # likely they were answering the pattern slot by mistake — reject for semester.
        if re.search(r"\b(19|20)\d{2}\b", raw) or "pattern" in raw or "endsem" in raw or "insem" in raw:
            dispatcher.utter_message(text="That looks like a year/pattern. Please tell me the semester like 'Sem 3' or '3'.")
            return {"semester": None}

        # accept "sem 3", "3", "3rd", "third"
        m = re.search(r"\b([1-9]|1[0-2])\b", raw)
        if m:
            sem_num = int(m.group(1))
            if 1 <= sem_num <= 12:
                return {"semester": f"Sem {sem_num}"}

        # words mapping
        word_map = {
            "first": 1, "second": 2, "third": 3, "fourth": 4,
            "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12
        }
        for word, num in word_map.items():
            if re.search(rf"\b{word}\b", raw):
                return {"semester": f"Sem {num}"}

        dispatcher.utter_message(text="I couldn't understand the semester. Please reply like 'Sem 3' or just '3'.")
        return {"semester": None}

    def validate_selected_subject(self, slot_value: Any, dispatcher: CollectingDispatcher,
                                  tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if not slot_value:
            dispatcher.utter_message(text="Which subject would you like me to analyze?")
            return {"selected_subject": None}
        val_norm = re.sub(r"\s+", " ", str(slot_value)).title()
        return {"selected_subject": val_norm}

    def validate_pattern(self, slot_value: Any, dispatcher: CollectingDispatcher,
                         tracker: Tracker, domain: Dict[Text, Any]) -> Dict[Text, Any]:
        if not slot_value:
            dispatcher.utter_message(text="Please provide the exam pattern or year (e.g., '2019' or '2015-2016').")
            return {"pattern": None}

        raw = str(slot_value).strip().lower()

        # Strict: accept a 4-digit year or year-range like '2015-2016' or '2015 to 2016'
        m_range = re.search(r"((?:19|20)\d{2})\s*[-–to]+\s*((?:19|20)\d{2})", raw)
        if m_range:
            y1, y2 = int(m_range.group(1)), int(m_range.group(2))
            if y1 <= y2:
                return {"pattern": f"{y1}-{y2}"}

        m_year = re.search(r"\b((?:19|20)\d{2})\b", raw)
        if m_year:
            return {"pattern": m_year.group(1)}

        # Accept phrases like 'insem 2019' or 'endsem 2019' by extracting the year
        m = re.search(r"(insem|endsem).*(\b(?:19|20)\d{2}\b)", raw)
        if m:
            return {"pattern": m.group(2)}

        # Otherwise ask for clarification
        dispatcher.utter_message(text="I couldn't extract a year from that. Please reply like '2019' or '2015-2016'.")
        return {"pattern": None}
    def name(self) -> Text:
        return "validate_paper_analysis_form"

    def validate_department(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Normalize department text and do a lightweight validation."""
        if not slot_value:
            dispatcher.utter_message(text="Please tell me the department (for example: Computer Engineering, IT).")
            return {"department": None}

        val = str(slot_value).strip()
        # Basic normalization: title case, remove extra whitespace
        val_norm = re.sub(r"\s+", " ", val).title()

        # Optional: a small whitelist to catch obvious typos (extend as needed)
        common_departments = {
            "Computer Engineering",
            "Information Technology",
            "Mechanical Engineering",
            "Civil Engineering",
            "Electrical Engineering",
            "Electronics",
        }
        if val_norm in common_departments:
            return {"department": val_norm}
        # Accept anything non-empty (user may have custom dept), but confirm back
        dispatcher.utter_message(text=f"Got department as '{val_norm}'. If that's correct, continue or re-type.")
        return {"department": val_norm}

    def validate_semester(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Accepts 'sem 3', '3', '3rd', 'third semester', etc. Normalizes to 'Sem X'."""
        if not slot_value:
            dispatcher.utter_message(text="Which semester is this for? (e.g., sem 3)")
            return {"semester": None}

        val = str(slot_value).strip().lower()
        # try to extract a number 1-12
        # common patterns: "sem 3", "3", "3rd", "third"
        # first, direct digits
        m = re.search(r"([1-9]|1[0-2])", val)
        if m:
            sem_num = int(m.group(1))
            sem_norm = f"Sem {sem_num}"
            return {"semester": sem_norm}

        # words to numbers map for basic cases
        word_map = {
            "first": 1, "second": 2, "third": 3, "fourth": 4,
            "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
            "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12
        }
        for word, num in word_map.items():
            if word in val:
                return {"semester": f"Sem {num}"}

        dispatcher.utter_message(text="I couldn't understand the semester. Please enter like 'Sem 3' or '3'.")
        return {"semester": None}

    def validate_selected_subject(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Normalize subject name. If empty, ask again."""
        if not slot_value:
            dispatcher.utter_message(text="Which subject would you like me to analyze?")
            return {"selected_subject": None}

        val = str(slot_value).strip()
        # simple normalization
        val_norm = re.sub(r"\s+", " ", val).title()

        # Optionally check against list of common subjects (extendable)
        common_subjects = {
            "Mathematics", "Physics", "Chemistry", "Computer Networks",
            "Data Structures", "Algorithms", "Operating Systems",
            "Database Management", "Software Engineering"
        }
        if val_norm in common_subjects:
            return {"selected_subject": val_norm}

        # Accept free-text subject but confirm
        dispatcher.utter_message(text=f"Subject recorded as '{val_norm}'. If that's correct, I'll proceed.")
        return {"selected_subject": val_norm}

    def validate_pattern(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """
        Pattern can be things like '2019 pattern', '2015-2016', 'pattern 2018', or simple year.
        Normalize to a readable form (e.g., '2019' or '2015-2016').
        """
        if not slot_value:
            dispatcher.utter_message(text="Please provide the exam pattern or year (e.g., '2019 pattern').")
            return {"pattern": None}

        val = str(slot_value).strip().lower()
        # find year or year range
        m_range = re.search(r"((?:19|20)\d{2})\s*[-–to]+\s*((?:19|20)\d{2})", val)
        if m_range:
            y1, y2 = m_range.group(1), m_range.group(2)
            if int(y1) <= int(y2):
                return {"pattern": f"{y1}-{y2}"}

        m_year = re.search(r"((?:19|20)\d{2})", val)
        if m_year:
            return {"pattern": m_year.group(1)}

        # catch words like '2019 pattern'
        m_word = re.search(r"pattern\s*(\d{4})", val)
        if m_word:
            return {"pattern": m_word.group(1)}

        # otherwise accept the raw text but ask for confirmation
        val_norm = re.sub(r"\s+", " ", val)
        dispatcher.utter_message(text=f"Pattern recorded as '{val_norm}'. If this is correct, I will proceed.")
        return {"pattern": val_norm}
    


