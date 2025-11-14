from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk.types import DomainDict
import os
import re
import spacy
import random
import PyPDF2
import logging
from docx import Document
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from rasa_sdk.executor import CollectingDispatcher
import matplotlib.pyplot as plt
nlp = spacy.load("en_core_web_sm")
logger = logging.getLogger(__name__)

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

class ActionHandleSmallTalk(Action):
    def name(self) -> Text:
        return "action_handle_small_talk"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get("text").lower()

        if "hello" in user_message:
            response = "Hey there! How can I assist you?"
        elif "how are you" in user_message:
            response = "I'm just a bot, but I'm here to help you!"
        else:
            response = "I love chatting! What do you need help with?"

        dispatcher.utter_message(text=response)
        return []


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

class ActionHandleFileUpload(Action):
    def name(self) -> Text:
        return "action_handle_file_upload"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        try:
            file_path = tracker.get_slot("uploaded_file")
            
            # Security checks
            if not os.path.exists(file_path):
                raise ValueError("File not found")
                
            if os.path.getsize(file_path) > Config.MAX_FILE_SIZE:
                raise ValueError(f"File exceeds {Config.MAX_FILE_SIZE//1024//1024}MB limit")
                
            if not any(file_path.lower().endswith(ext) for ext in Config.ALLOWED_EXTENSIONS):
                raise ValueError("Unsupported file type")
            
            return [SlotSet("uploaded_file", file_path)]

        except Exception as e:
            logger.error(f"File upload failed: {str(e)}")
            dispatcher.utter_message(text=f"Upload error: {str(e)}")
            return [SlotSet("uploaded_file", None)]

class ActionAnalyzeQuestionPaper(Action):
    def name(self) -> Text:
        return "action_analyze_question_paper"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        file_path = tracker.get_slot("uploaded_file")
        
        try:
            text = self._extract_text(file_path)
            questions = self._process_text(text)
            
            # New: Obtain semantic clusters and generate a vertical bar chart
            cluster_labels = self._get_cluster_labels(questions)
            image_path = self._plot_question_clusters(questions, cluster_labels, top_n=10)
            
            analysis = {
                "topics": self._identify_topics(questions),
                "frequent_questions": self._find_frequent_questions(questions),
                "difficulty": self._estimate_difficulty(questions),
                "question_types": self._categorize_question_types(questions),
                "cluster_plot": image_path
            }
            
            dispatcher.utter_message(text=self._format_analysis(analysis))
            # If your channel supports images, send the plot image as well.
            dispatcher.utter_message(image=image_path)
            return [SlotSet("analysis_results", analysis)]

        except Exception as e:
            logger.exception("Analysis failed")
            dispatcher.utter_message(text=f"Analysis error: {str(e)}")
            return []

    def _extract_text(self, file_path: Text) -> Text:
        """Extract text from supported file types"""
        try:
            if file_path.endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    return " ".join([page.extract_text() for page in reader.pages])
            elif file_path.endswith('.docx'):
                doc = Document(file_path)
                return " ".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise RuntimeError(f"Text extraction failed: {str(e)}")

    def _process_text(self, text: Text) -> List[Text]:
        """Clean and split questions"""
        text = re.sub(r'[^\w\s.?]', '', text)
        return [q.strip() for q in re.split(r'(?:Q\d+\.|Question\s+\d+:|\(\d+\)\s*)', text) if q.strip()]

    def _identify_topics(self, questions: List[Text]) -> List[Text]:
        """LDA Topic Modeling"""
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(questions)
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(dtm)
        return [", ".join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-3:]]) 
                for topic in lda.components_]

    def _find_frequent_questions(self, questions: List[Text]) -> List[Text]:
        """Semantic clustering to group similar questions and return representative questions"""
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        # Clean and prepare question texts
        question_texts = [self._clean_question_text(q) for q in questions]
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(question_texts, convert_to_tensor=False)
        cosine_sim = cosine_similarity(embeddings)
        
        threshold = 0.8
        n = len(question_texts)
        visited = [False] * n
        clusters = []
        for i in range(n):
            if not visited[i]:
                cluster = [i]
                visited[i] = True
                for j in range(i+1, n):
                    if not visited[j] and cosine_sim[i][j] >= threshold:
                        cluster.append(j)
                        visited[j] = True
                clusters.append(cluster)
        # Pick representative question from each cluster
        representatives = []
        for cluster in clusters:
            representatives.append(question_texts[cluster[0]])
        return representatives

    def _estimate_difficulty(self, questions: List[Text]) -> Text:
        """Heuristic difficulty estimation"""
        avg_length = np.mean([len(q.split()) for q in questions])
        return "Advanced" if avg_length > 50 else "Intermediate" if avg_length > 25 else "Basic"

    def _categorize_question_types(self, questions: List[Text]) -> Dict[Text, int]:
        """Question type classification"""
        patterns = {
            'Definition': r'define|what is|explain',
            'Calculation': r'calculate|solve|compute|formula',
            'Problem Solving': r'prove|demonstrate|solve the problem',
            'Comparison': r'compare|contrast|difference between',
            'Enumeration': r'list|name|give examples'
        }
        return {q_type: sum(1 for q in questions if re.search(pattern, q, re.IGNORECASE))
                for q_type, pattern in patterns.items()}

    def _format_analysis(self, analysis: Dict) -> Text:
        """Generate formatted report"""
        return (
            f"📊 Analysis Report:\n\n"
            f"🔍 Top Topics:\n{chr(10).join(analysis['topics'])}\n\n"
            f"📌 Frequent Questions:\n{chr(10).join(analysis['frequent_questions'])}\n\n"
            f"📈 Difficulty: {analysis['difficulty']}\n\n"
            f"🧩 Question Types:\n{chr(10).join(f'- {k}: {v}' for k,v in analysis['question_types'].items())}\n\n"
            f"🖼 Cluster Plot saved at: {analysis['cluster_plot']}"
        )
    
    # ----------------------- NEW HELPER METHODS ---------------------------
    def _clean_question_text(self, text: Text) -> Text:
        """Removes stray isolated numbers and extra spaces from the question text"""
        cleaned = re.sub(r"(?<=\s)\d{1,2}(?=\s)", " ", text)
        cleaned = re.sub(r"^\d{1,2}\s+", "", cleaned)
        cleaned = re.sub(r"\s+\d{1,2}$", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    def _get_cluster_labels(self, questions: List[Text]) -> List[int]:
        """Compute cluster labels for questions using sentence embeddings"""
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        question_texts = [self._clean_question_text(q) for q in questions]
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(question_texts, convert_to_tensor=False)
        cosine_sim = cosine_similarity(embeddings)
        
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
                    if not visited[j] and cosine_sim[i][j] >= threshold:
                        labels[j] = cluster_id
                        visited[j] = True
                cluster_id += 1
        return labels

    def _plot_question_clusters(self, questions: List[Text], cluster_labels: List[int], top_n=10) -> Text:
        """Generate a vertical bar chart of the top clusters and save it as an image"""
        from collections import Counter
        
        cluster_counter = Counter(cluster_labels)
        top_clusters = cluster_counter.most_common(top_n)
        clusters = [f"Cluster {label}" for label, _ in top_clusters]
        frequencies = [count for _, count in top_clusters]
        
        plt.figure(figsize=(12, 6))
        plt.bar(clusters, frequencies, color="skyblue")
        plt.xlabel("Clusters (Grouped Question Types)")
        plt.ylabel("Frequency")
        plt.title("Top Frequently Asked Question Types")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        image_path = "question_clusters.png"
        plt.savefig(image_path)
        plt.close()
        return image_path

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

    def validate_subject(
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

    def validate_duration(
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