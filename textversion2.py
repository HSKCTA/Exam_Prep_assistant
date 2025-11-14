import pdfplumber
import re
from collections import Counter
import matplotlib.pyplot as plt

# For clustering similar questions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

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
    """
    Remove stray isolated numbers (likely artifacts) from the question text.
    """
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
    all_questions = []
    for pdf_path in pdf_paths:
        clean_text = extract_clean_text(pdf_path)
        questions = extract_questions_and_marks(clean_text)
        all_questions.extend(questions)
    return all_questions

def cluster_similar_questions(questions, similarity_threshold=0.8):
    """
    Cluster questions using TF-IDF and Agglomerative Clustering.
    Returns a list of cluster labels corresponding to the input questions.
    """
    # Get question texts and clean them.
    question_texts = [clean_question_text(q["question"]) for q in questions]
    
    # Vectorize using TF-IDF.
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(question_texts)
    
    # Use Agglomerative Clustering with cosine metric.
    clustering = AgglomerativeClustering(metric='cosine', linkage='average',
                                         distance_threshold=1 - similarity_threshold,
                                         n_clusters=None)
    labels = clustering.fit_predict(X.toarray())
    return labels


def plot_most_frequent_clusters(questions, labels, top_n=10):
    """
    Group questions by cluster labels and plot the frequency (vertical bar chart).
    The representative question for each cluster is taken as the first question in that cluster.
    """
    # Count frequency of each cluster label.
    cluster_counter = Counter(labels)
    
    # Create a dictionary mapping cluster label to a representative question.
    cluster_representative = {}
    for q, label in zip(questions, labels):
        if label not in cluster_representative:
            cluster_representative[label] = clean_question_text(q["question"])
    
    # Get the top clusters.
    top_clusters = cluster_counter.most_common(top_n)
    
    clusters = [f"Cluster {label}" for label, _ in top_clusters]
    frequencies = [count for _, count in top_clusters]
    
    # Optionally, you can print representative texts for each top cluster:
    for label, count in top_clusters:
        print(f"Cluster {label} (Frequency: {count}): {cluster_representative[label]}")
    
    plt.figure(figsize=(10, 6))
    plt.bar(clusters, frequencies, color="skyblue")
    plt.xlabel("Clusters (Representative Question)")
    plt.ylabel("Frequency")
    plt.title("Top Clusters of Similar Questions")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


pdf_files = [r"C:\Users\hites\Downloads\SE Endsem 1.pdf",r"C:\Users\hites\Downloads\SE Endsem 2.pdf" ,r"C:\Users\hites\Downloads\SE Endsem 2024 paper.pdf" ]
extracted_questions = process_multiple_papers(pdf_files)
labels = cluster_similar_questions(extracted_questions, similarity_threshold=0.8)
plot_most_frequent_clusters(extracted_questions, labels, top_n=10)