

import json
import re
import time
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import NLTKTextSplitter
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import os
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import json
import ast
from pathlib import Path            
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from xlsxwriter import Workbook
os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    accuracy: float
    semantic_similarity: float
    completeness: float
    relevance: float
    syntax_correctness: float
    overall_score: float
    details: Dict[str, Any]

class RAGEvaluator:
    """
    Comprehensive RAG system evaluator for PlantUML/UML documentation
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the evaluator with a sentence transformer model"""
        self.sentence_model = SentenceTransformer(model_name)
        self.ground_truth = self._load_ground_truth()
        
    def _load_ground_truth(self) -> List[Dict]:
        """
        Ground truth dataset for PlantUML/UML documentation evaluation
        """
        return [
            {
                "id": "plantuml_basic_01",
                "question": "How do you start and end a PlantUML class diagram?",
                "expected_answer": "Use @startuml at the beginning and @enduml at the end of the diagram",
                "key_concepts": ["@startuml", "@enduml", "class diagram"],
                "difficulty": "easy",
                "category": "syntax"
            },
            {
                "id": "plantuml_basic_02", 
                "question": "What is the syntax to define a class in PlantUML?",
                "expected_answer": "Use 'class ClassName' or just 'ClassName' to define a class",
                "key_concepts": ["class", "class definition", "syntax"],
                "difficulty": "easy",
                "category": "syntax"
            },
            {
                "id": "plantuml_relationship_01",
                "question": "How do you represent inheritance in PlantUML class diagrams?",
                "expected_answer": "Use '--|>' or '<|--' to represent inheritance relationship",
                "key_concepts": ["inheritance", "--|>", "<|--", "relationship"],
                "difficulty": "medium",
                "category": "relationships"
            },
            {
                "id": "plantuml_relationship_02",
                "question": "What symbols are used for composition and aggregation in PlantUML?",
                "expected_answer": "Composition uses '*--' or '--*', aggregation uses 'o--' or '--o'",
                "key_concepts": ["composition", "aggregation", "*--", "o--"],
                "difficulty": "medium", 
                "category": "relationships"
            },
            {
                "id": "plantuml_attributes_01",
                "question": "How do you add attributes to a class in PlantUML?",
                "expected_answer": "Use class ClassName { attribute_name : type } or ClassName : attribute_name",
                "key_concepts": ["attributes", "class members", "type specification"],
                "difficulty": "medium",
                "category": "class_structure"
            },
            {
                "id": "plantuml_methods_01",
                "question": "What's the syntax for adding methods to a class in PlantUML?",
                "expected_answer": "Use method_name() or method_name(parameters) : return_type inside class definition",
                "key_concepts": ["methods", "parameters", "return type", "class members"],
                "difficulty": "medium",
                "category": "class_structure"
            },
            {
                "id": "plantuml_visibility_01",
                "question": "How do you specify visibility modifiers in PlantUML?",
                "expected_answer": "Use + for public, - for private, # for protected, ~ for package",
                "key_concepts": ["visibility", "public", "private", "protected", "package"],
                "difficulty": "medium",
                "category": "class_structure"
            },
            {
                "id": "plantuml_advanced_01",
                "question": "How do you create abstract classes and interfaces in PlantUML?",
                "expected_answer": "Use 'abstract class ClassName' for abstract classes and 'interface InterfaceName' for interfaces",
                "key_concepts": ["abstract", "interface", "abstract class"],
                "difficulty": "hard",
                "category": "advanced"
            },
            {
                "id": "plantuml_advanced_02",
                "question": "What's the syntax for adding notes to PlantUML diagrams?",
                "expected_answer": "Use 'note left of', 'note right of', 'note top of', 'note bottom of' followed by class name",
                "key_concepts": ["notes", "annotations", "documentation"],
                "difficulty": "medium",
                "category": "documentation"
            },
            {
                "id": "uml_principles_01",
                "question": "What are the main types of UML diagrams?",
                "expected_answer": "Structural diagrams (class, object, component, deployment) and behavioral diagrams (sequence, activity, use case, state)",
                "key_concepts": ["UML diagrams", "structural", "behavioral", "diagram types"],
                "difficulty": "medium",
                "category": "uml_theory"
            },
            {
                "id": "uml_principles_02",
                "question": "What does multiplicity mean in UML relationships?",
                "expected_answer": "Multiplicity specifies the number of instances that can participate in a relationship, like 1, 0..1, 1..*, or *",
                "key_concepts": ["multiplicity", "cardinality", "relationship", "instances"],
                "difficulty": "hard",
                "category": "uml_theory"
            },
            {
                "id": "plantuml_styling_01",
                "question": "How do you change colors in PlantUML diagrams?",
                "expected_answer": "Use skinparam or add color specifications like 'class ClassName #LightBlue'",
                "key_concepts": ["colors", "styling", "skinparam", "appearance"],
                "difficulty": "medium",
                "category": "styling"
            },
            {
                "id": "plantuml_package_01",
                "question": "How do you organize classes into packages in PlantUML?",
                "expected_answer": "Use 'package PackageName { }' to group classes or 'package PackageName' with end",
                "key_concepts": ["packages", "namespaces", "organization", "grouping"],
                "difficulty": "medium",
                "category": "organization"
            },
            {
                "id": "plantuml_stereotypes_01",
                "question": "What are stereotypes in PlantUML and how do you use them?",
                "expected_answer": "Stereotypes are labels that extend UML elements, use <<stereotype>> syntax like class ClassName <<Entity>>",
                "key_concepts": ["stereotypes", "labels", "extensions", "<<>>"],
                "difficulty": "hard",
                "category": "advanced"
            },
            {
                "id": "plantuml_association_01",
                "question": "How do you add labels to associations in PlantUML?",
                "expected_answer": "Use ClassA -- ClassB : label or ClassA --> ClassB : label",
                "key_concepts": ["associations", "labels", "relationships", "naming"],
                "difficulty": "medium",
                "category": "relationships"
            }

            
        ]
    
    def evaluate_response(self, question: str, response: str, chunks: List[str]) -> EvaluationResult:
        """
        Evaluate a RAG system response against ground truth
        
        Args:
            question: The input question
            response: The RAG system's response
            chunks: The retrieved chunks used to generate the response
            
        Returns:
            EvaluationResult with comprehensive metrics
        """
        # Find matching ground truth
        ground_truth_item = self._find_ground_truth(question)
        if not ground_truth_item:
            return EvaluationResult(0, 0, 0, 0, 0, 0, {"error": "No ground truth found"})
        
        # Calculate metrics
        accuracy = self._calculate_accuracy(response, ground_truth_item)
        semantic_similarity = self._calculate_semantic_similarity(response, ground_truth_item["expected_answer"])
        completeness = self._calculate_completeness(response, ground_truth_item["key_concepts"])
        relevance = self._calculate_relevance(response, chunks, ground_truth_item)
        syntax_correctness = self._calculate_syntax_correctness(response, ground_truth_item)
        
        # Calculate overall score
        overall_score = (accuracy * 0.25 + semantic_similarity * 0.25 + 
                        completeness * 0.2 + relevance * 0.15 + syntax_correctness * 0.15)
        
        return EvaluationResult(
            accuracy=accuracy,
            semantic_similarity=semantic_similarity,
            completeness=completeness,
            relevance=relevance,
            syntax_correctness=syntax_correctness,
            overall_score=overall_score,
            details={
                "question_id": ground_truth_item["id"],
                "difficulty": ground_truth_item["difficulty"],
                "category": ground_truth_item["category"],
                "key_concepts_found": self._find_key_concepts(response, ground_truth_item["key_concepts"]),
                "chunk_relevance_scores": self._score_chunk_relevance(chunks, ground_truth_item)
            }
        )
    
    def _find_ground_truth(self, question: str) -> Dict:
        """Find the best matching ground truth item for a question"""
        # Simple approach: find by highest similarity
        best_match = None
        best_score = 0
        
        question_embedding = self.sentence_model.encode([question])
        
        for gt_item in self.ground_truth:
            gt_embedding = self.sentence_model.encode([gt_item["question"]])
            similarity = cosine_similarity(question_embedding, gt_embedding)[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_match = gt_item
        
        return best_match if best_score > 0.7 else None
    
    def _calculate_accuracy(self, response: str, ground_truth: Dict) -> float:
        """Calculate accuracy using keyword matching and semantic similarity"""
        expected = ground_truth["expected_answer"].lower()
        response_lower = response.lower()
        
        # Keyword matching
        keywords = [concept.lower() for concept in ground_truth["key_concepts"]]
        keyword_matches = sum(1 for keyword in keywords if keyword in response_lower)
        keyword_score = keyword_matches / len(keywords) if keywords else 0
        
        # Semantic similarity
        expected_embedding = self.sentence_model.encode([expected])
        response_embedding = self.sentence_model.encode([response])
        semantic_score = cosine_similarity(expected_embedding, response_embedding)[0][0]
        
        return (keyword_score * 0.4 + semantic_score * 0.6)
    
    def _calculate_semantic_similarity(self, response: str, expected: str) -> float:
        """Calculate semantic similarity between response and expected answer"""
        response_embedding = self.sentence_model.encode([response])
        expected_embedding = self.sentence_model.encode([expected])
        return cosine_similarity(response_embedding, expected_embedding)[0][0]
    
    def _calculate_completeness(self, response: str, key_concepts: List[str]) -> float:
        """Calculate how many key concepts are covered in the response"""
        response_lower = response.lower()
        covered_concepts = sum(1 for concept in key_concepts if concept.lower() in response_lower)
        return covered_concepts / len(key_concepts) if key_concepts else 0
    
    def _calculate_relevance(self, response: str, chunks: List[str], ground_truth: Dict) -> float:
        """Calculate relevance of retrieved chunks to the question and answer"""
        if not chunks:
            return 0
        
        question_embedding = self.sentence_model.encode([ground_truth["question"]])
        chunk_embeddings = self.sentence_model.encode(chunks)
        
        similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
        return float(np.mean(similarities))
    
    def _calculate_syntax_correctness(self, response: str, ground_truth: Dict) -> float:
        """Calculate syntax correctness for PlantUML-specific responses"""
        if ground_truth["category"] != "syntax":
            return 1.0  # Not applicable for non-syntax questions
        
        # Check for common PlantUML syntax patterns
        syntax_patterns = [
            r'@startuml', r'@enduml', r'class\s+\w+', r'-->', r'<--', r'--|>', r'<|--',
            r'\+\s*\w+', r'-\s*\w+', r'#\s*\w+', r'~\s*\w+', r'<<\w+>>', r'package\s+\w+'
        ]
        
        found_patterns = sum(1 for pattern in syntax_patterns if re.search(pattern, response, re.IGNORECASE))
        expected_patterns = len([p for p in syntax_patterns if re.search(p, ground_truth["expected_answer"], re.IGNORECASE)])
        
        if expected_patterns == 0:
            return 1.0
        
        return min(found_patterns / expected_patterns, 1.0)
    
    def _find_key_concepts(self, response: str, key_concepts: List[str]) -> List[str]:
        """Find which key concepts are present in the response"""
        response_lower = response.lower()
        return [concept for concept in key_concepts if concept.lower() in response_lower]
    
    def _score_chunk_relevance(self, chunks: List[str], ground_truth: Dict) -> List[float]:
        """Score the relevance of each chunk to the ground truth"""
        if not chunks:
            return []
        
        question_embedding = self.sentence_model.encode([ground_truth["question"]])
        chunk_embeddings = self.sentence_model.encode(chunks)
        
        similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
        return similarities.tolist()
    
    def evaluate_batch(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate multiple test cases and return aggregate results
        
        Args:
            test_cases: List of {"question": str, "response": str, "chunks": List[str]}
            
        Returns:
            Dictionary with aggregate metrics and detailed results
        """
        results = []
        
        for test_case in test_cases:
            result = self.evaluate_response(
                test_case["question"],
                test_case["response"], 
                test_case["chunks"]
            )
            results.append(result)
        
        # Calculate aggregate metrics
        metrics = ["accuracy", "semantic_similarity", "completeness", "relevance", "syntax_correctness", "overall_score"]
        aggregates = {}
        
        for metric in metrics:
            values = [getattr(result, metric) for result in results]
            aggregates[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        return {
            "aggregate_metrics": aggregates,
            "individual_results": results,
            "total_questions": len(test_cases),
            "categories": self._analyze_by_category(results),
            "difficulty_analysis": self._analyze_by_difficulty(results),
            "embedding_model": test_cases[0]["embedding_model"],
            "splitter_config": test_cases[0]["splitter_config"],
            "retriever": test_cases[0]["retriever"],
            "retrieval_time": test_cases[0]["retrieval_time"],
            "num_chunks": test_cases[0]["num_chunks"]
        }
    
    def _analyze_by_category(self, results: List[EvaluationResult]) -> Dict:
        """Analyze results by category"""
        categories = {}
        for result in results:
            if "category" in result.details:
                category = result.details["category"]
                if category not in categories:
                    categories[category] = []
                categories[category].append(result.overall_score)
        
        return {cat: {"mean": np.mean(scores), "count": len(scores)} 
                for cat, scores in categories.items()}
    
    def _analyze_by_difficulty(self, results: List[EvaluationResult]) -> Dict:
        """Analyze results by difficulty level"""
        difficulties = {}
        for result in results:
            if "difficulty" in result.details:
                difficulty = result.details["difficulty"]
                if difficulty not in difficulties:
                    difficulties[difficulty] = []
                difficulties[difficulty].append(result.overall_score)
        
        return {diff: {"mean": np.mean(scores), "count": len(scores)} 
                for diff, scores in difficulties.items()}
    
    def generate_report(self, evaluation_results: Dict) -> str:
        """Generate a human-readable evaluation report"""
        report = "RAG System Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Overall metrics
        report += "Overall Performance:\n"
        report += "-" * 20 + "\n"
        for metric, values in evaluation_results["aggregate_metrics"].items():
            report += f"{metric.replace('_', ' ').title()}: {values['mean']:.3f} (±{values['std']:.3f})\n"
        
        # Category analysis
        report += "\nPerformance by Category:\n"
        report += "-" * 25 + "\n"
        for category, stats in evaluation_results["categories"].items():
            report += f"{category.replace('_', ' ').title()}: {stats['mean']:.3f} ({stats['count']} questions)\n"
        
        # Difficulty analysis
        report += "\nPerformance by Difficulty:\n"
        report += "-" * 26 + "\n"
        for difficulty, stats in evaluation_results["difficulty_analysis"].items():
            report += f"{difficulty.title()}: {stats['mean']:.3f} ({stats['count']} questions)\n"
        
        return report
    def export_evaluation_to_excel(self,evaluation_results: Dict[str, Any],output_path: str | Path = "evaluation_report.xlsx",) -> Path:
        """
        Write the evaluate_batch() dictionary to an Excel workbook whose sheets
        match the required column specs in the prompt.
        """
        output_path = Path(output_path)

        # ------------------------------------------------------------------ #
        # Common metadata (re‑used across several sheets)
        # ------------------------------------------------------------------ #
        meta = {
            "Embedding Model" : evaluation_results["embedding_model"],
            "Splitter Config" : evaluation_results["splitter_config"],
            "Retriever"       : evaluation_results["retriever"],
            "Retrieval Time"  : evaluation_results["retrieval_time"],
            "Num Chunks"      : evaluation_results["num_chunks"],
            "Total Questions" : evaluation_results["total_questions"],
        }

        # ------------------------------------------------------------------ #
        # 1. Overall Metrics sheet
        # ------------------------------------------------------------------ #
        overall_df = (
            pd.DataFrame(evaluation_results["aggregate_metrics"])
            .T
            .rename_axis("metric")
            .reset_index()
            .rename(columns={
                "metric": "metric",
                "mean"  : "mean",
                "std"   : "std",
                "min"   : "min",
                "max"   : "max",
            })
        )
        # replicate the meta columns for every row
        for k, v in meta.items():
            overall_df[k] = v
        # reorder to the requested column order
        overall_df = overall_df[
            [
                "Embedding Model", "Splitter Config", "Retriever",
                "Retrieval Time", "Num Chunks", "Total Questions",
                "metric", "mean", "std", "min", "max",
            ]
        ]

        # ------------------------------------------------------------------ #
        # 2. By‑Category sheet
        # ------------------------------------------------------------------ #
        category_df = (
            pd.DataFrame(evaluation_results["categories"])
            .T
            .rename_axis("category")
            .reset_index()
            .rename(columns={"mean": "mean", "count": "count"})
        )
        for k in ("Embedding Model", "Splitter Config", "Retriever"):
            category_df[k] = meta[k]
        category_df = category_df[
            ["Embedding Model", "Splitter Config", "Retriever",
            "category", "mean", "count"]
        ]

        # ------------------------------------------------------------------ #
        # 3. By‑Difficulty sheet
        # ------------------------------------------------------------------ #
        difficulty_df = (
            pd.DataFrame(evaluation_results["difficulty_analysis"])
            .T
            .rename_axis("difficulty")
            .reset_index()
            .rename(columns={"mean": "mean", "count": "count"})
        )
        for k in ("Embedding Model", "Splitter Config", "Retriever"):
            difficulty_df[k] = meta[k]
        difficulty_df = difficulty_df[
            ["Embedding Model", "Splitter Config", "Retriever",
            "difficulty", "mean", "count"]
        ]

        # ------------------------------------------------------------------ #
        # 4. Individual Results sheet
        # ------------------------------------------------------------------ #
        individual_results = [
            (r.__dict__ if hasattr(r, "__dict__") else r)
            for r in evaluation_results["individual_results"]
        ]
        individual_df = pd.DataFrame(individual_results)[
            [
                "semantic_similarity",
                "completeness",
                "relevance",
                "syntax_correctness",
                "overall_score",
                "details",
            ]
        ]

        # ------------------------------------------------------------------ #
        # Write to Excel
        # ------------------------------------------------------------------ #
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            overall_df.to_excel(writer, sheet_name="Overall Metrics", index=False)
            category_df.to_excel(writer, sheet_name="By Category", index=False)
            difficulty_df.to_excel(writer, sheet_name="By Difficulty", index=False)
            individual_df.to_excel(writer, sheet_name="Individual Results", index=False)

            # autosize columns for readability
            for sheet_name, df in {
                "Overall Metrics": overall_df,
                "By Category": category_df,
                "By Difficulty": difficulty_df,
                "Individual Results": individual_df,
            }.items():
                worksheet = writer.sheets[sheet_name]
                for i, col in enumerate(df.columns):
                    width = max(df[col].astype(str).str.len().max(), len(col)) + 2
                    worksheet.set_column(i, i, width)

        print(f" Excel report saved to {output_path.resolve()}")
        return output_path

class RAGSystemConfig:
    """
    Configuration class for RAG system testing
    """
    
    def __init__(self, vector_store, retriever):
        self.vector_store = vector_store
        self.retriever = retriever
        
        self.client =  ChatAnthropic(
            model="claude-3-haiku-20240307",  # or "claude-3-sonnet-20240229", etc.
            temperature=0.2,
            max_tokens=500
        )

    def clean_chunk(self,text):
        lines = text.splitlines()
        cleaned_lines = [
            line.strip()
            for line in lines
            if line.strip() and line.strip() != '•'
        ]
        return " ".join(cleaned_lines)

    def generate_results(self,question):
        start_time = time.time()
        docs = self.retriever.get_relevant_documents(question)
        retrieval_time = time.time() - start_time
        chunks = [doc.page_content for doc in docs]
        context = [self.clean_chunk(chunk) for chunk in chunks]

        system_prompt = """ You are an expert in PlantUML and UML class diagrams. You have a deep understanding of how to create and interpret UML class diagrams using PlantUML syntax. Your responses should be concise, accurate, and focused on providing clear guidance for creating class diagrams. """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"**QUESTION** {question}\n\n**CONTEXT**\n" + "\n\n".join(context)),
        ]

        response = self.client.invoke(messages)
        return response, context, retrieval_time

        
       

def clean_text( text: str) -> str:
    """Clean and normalize text content"""
    import re
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere
    text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\[\]\{\}]', '', text)
    return text.strip()
    
def load_documents() -> List[Document]:
    """Load and clean all documents"""
    print("Loading documents...")
    
    # Load PDF
    pdf_loader = PyPDFLoader("data/PlantUML_Language_Reference_Guide_en.pdf")
    pdf_data = pdf_loader.load()
    for doc in pdf_data:
        doc.page_content = clean_text(doc.page_content)
    
    # Load web content
    web_urls = [
        "https://plantuml.com/class-diagram",
        "https://www.geeksforgeeks.org/system-design/unified-modeling-language-uml-class-diagrams/"
    ]
    
    web_data = []
    for url in web_urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)
            web_data.extend(docs)
        except Exception as e:
            print(f"Error loading {url}: {e}")
    
    # Add source metadata
    for doc in pdf_data:
        doc.metadata["source"] = "pdf"
    for doc in web_data:
        doc.metadata["source"] = "web"
        
    return pdf_data + web_data

def get_embedding_models() -> Dict[str, Any]:
    """Define different embedding models to test"""
    return {
        "all-MiniLM-L6-v2": HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        "all-mpnet-base-v2": HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
        "multi-qa-MiniLM-L6-cos-v1": HuggingFaceEmbeddings(
            model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        ),
        "all-distilroberta-v1": HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-distilroberta-v1"
        ),
        "paraphrase-multilingual-MiniLM-L12-v2": HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

      
    }
def get_text_splitter_configs() -> Dict[str, object]:
    """Define different text splitting configurations  https://arxiv.org/abs/2505.21700?"""
    return {
        "default": RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=20, separators=["\n\n", "\n", ". ", " ", ""]
        ),
        "large_chunks": RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        ),
        "semantic": NLTKTextSplitter(chunk_size=300, chunk_overlap=20),
        
    }

def create_vector_store( documents: List[Document], embedding_model, 
                          text_splitter,emb_name:str ,split_name: str, store_type: str = "chroma") -> Any:
        """Create vector store with given configuration"""
        
        # Create text splitter
     
        
        # Split documents
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        if store_type == "chroma":
            client = chromadb.Client()                   # guarantees blank slate
            collection_name = f"{emb_name}_{split_name}".replace("/", "_")

            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embedding_model,
                client=client,
                collection_name=collection_name,    
            )
        elif store_type == "faiss":
            vector_store = FAISS.from_documents(
                documents=splits,
                embedding=embedding_model
            )
        
        return vector_store, len(splits)
        
def create_retrievers( vector_store, documents: List[Document]) -> Dict[str, Any]:
    """Create different types of retrievers"""
    retrievers = {}
    k_values = [ 3]
    thresholds = [0.5]

    for k in k_values:
        retrievers[f"similarity_k{k}"] = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
        retrievers[f"mmr_k{k}"] = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "fetch_k": max(k + 5, 10)}
        )

    for t in thresholds:
        retrievers[f"threshold_{t}"] = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": t, "k": 10}
        )
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    retrievers["bm25_k5"] = bm25_retriever

    # Ensemble (example: 0.5 BM25 + 0.5 vector)
    retrievers["ensemble"] = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_store.as_retriever()],
        weights=[0.5, 0.5]
    )
    
    return retrievers
    
def run_comprehensive_test():
    """Run comprehensive testing of different combinations"""
    print("Starting comprehensive retrieval testing...")
    

    documents = load_documents()
    embedding_models = get_embedding_models()
    text_splitter_configs = get_text_splitter_configs()   
     
    for emb_name, embedding_model in embedding_models.items():
        emb_results = []
            
        print(f"\nTesting embedding model: {emb_name}")
        
        for split_name, split_config in text_splitter_configs.items():
            
                
            print(f"  Testing splitter config: {split_name}")
            
            try:
                # Create vector store
                vector_store, num_chunks = create_vector_store(
                    documents=documents, embedding_model=embedding_model, text_splitter = split_config, split_name=split_name, emb_name=emb_name
                )
                
                # Create retrievers
                retrievers = create_retrievers(vector_store, documents)
                
                for ret_name, retriever in retrievers.items():
                    
                        
                    print(f"    Testing retriever: {ret_name}")
                    resultsToEvaluate = []
                    rag_system= RAGSystemConfig( vector_store =  vector_store, retriever= retriever )
                    tests_cases = RAGEvaluator()._load_ground_truth()
                    for test_case in tests_cases:
                        question = test_case["question"]
                        response,context, retrieval_time = rag_system.generate_results(question)
                        generated_result = {
                            "question": question,
                            "response": response.content,
                            "chunks": context,
                            "embedding_model": emb_name,
                            "splitter_config": split_name,
                            "retriever": ret_name,
                            "retrieval_time": retrieval_time,
                            "num_chunks": num_chunks
                        }
                        resultsToEvaluate.append(generated_result)
                    example_usage(resultsToEvaluate)
                    emb_results.append(generated_result)

                    
            except Exception as e:
                print(f"    Error with combination {emb_name}/{split_name}: {e}")

        try:
            data = emb_results
        except (SyntaxError, ValueError) as exc:
            raise SystemExit(f"Could not parse emb_results: {exc}")
        path = "RunningResults_" + emb_name + ".json"
        output_path = Path(path)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
 

    

def example_usage(test_cases):
    
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Example test cases
    """
    test_cases =[
  {
    "question": "How do you start and end a PlantUML class diagram?",
    "response": "To start and end a PlantUML class diagram, you can use the following syntax:\n\nStart the class diagram:\n```\n@startuml\n```\n\nDefine your classes and relationships:\n```\nclass ClassName {\n  - attribute1: type\n  - attribute2: type\n  + method1(param1: type): returnType\n  + method2(param1: type, param2: type): returnType\n}\n\nclass AnotherClassName {\n  # protectedAttribute: type\n  ~ packageAttribute: type\n  + publicMethod(param: type): returnType\n}\n\nrelationship_type ClassName \"multiplicity\" --> \"multiplicity\" AnotherClassName\n```\n\nEnd the class diagram:\n```\n@enduml\n```\n\nThe key points are:\n\n1. Start the diagram with `@startuml`.\n2. Define your classes using the `class` keyword, with attributes and methods.\n3. Specify the visibility of class members using the `-`, `#`, `~`, and `+` prefixes for private, protected, package, and public, respectively.\n4. Define relationships between classes using the appropriate relationship types (e.g., `-->`).\n5. End the diagram with `@enduml`.\n\nThis basic structure allows you to create UML class diagrams using the PlantUML syntax. You can then use the PlantUML renderer to generate the diagram image.",
    "chunks": [
      "UML Class Diagram Design Guide 1. Introduction This document is a comprehensive knowledge base for designing high-quality UML class diagrams from system  descriptions.  It  defines  key  UML  concepts,  class  design  principles,  PlantUML  examples, relationship  types  and  their  correct  usage,  design  patterns,  and  object-oriented  principles  like encapsulation and polymorphism. 2. UML Class Diagram Concepts 2.1. Class",
      "class Scene { -location: String -shootDate: Date } class Actor { -name: String } class Prop { -name: String } class Production { -title: String } Director \"1\" --> \"1..*\" Scene : manages Scene \"1\" --> \"1..*\" Actor : involves Scene \"1\" --> \"0..*\" Prop : uses Actor \"1\" --> \"*\" Scene : scheduled in Production \"1\" *-- \"1..*\" Prop : owns @enduml 5",
      "different ways. Implementation: Use interfaces or abstract base classes with overridden methods. abstract class Shape { +draw(): void } class Circle extends Shape { 3"
    ]
  },
  {
    "question": "What is the syntax to define a class in PlantUML?",
    "response": "The syntax to define a class in PlantUML is as follows:\n\n```\nclass <class_name> {\n  [+ | - | # | ~] <attribute_name>: <attribute_type>\n  [+ | - | # | ~] <method_name>(<parameter_list>): <return_type>\n}\n```\n\nHere's a breakdown of the syntax:\n\n- `class <class_name>`: This defines the class with the specified `<class_name>`.\n- `{`: This opens the class definition block.\n- `[+ | - | # | ~] <attribute_name>: <attribute_type>`: This defines an attribute of the class. The access modifier can be one of the following:\n  - `+`: public\n  - `-`: private\n  - `#`: protected\n  - `~`: package-private (default)\n- `[+ | - | # | ~] <method_name>(<parameter_list>): <return_type>`: This defines a method of the class. The access modifier can be the same as for attributes.\n- `}`: This closes the class definition block.\n\nHere's an example of a class definition in PlantUML:\n\n```\nclass Person {\n  + name: String\n  + age: int\n  + email: String\n  + getId(): String\n  - calculateAge(): int\n}\n```\n\nThis defines a `Person` class with three attributes (`name`, `age`, and `email`) and two methods (`getId()` and `calculateAge()`). The `name`, `age`, and `email` attributes are public, while the `calculateAge()` method is private.",
    "chunks": [
      "9. Summary Table Concept Definition Example UML Syntax When to Use Class Blueprint for objects class Person {} Represent structure and behavior of entities Association Structural link between classes A --> B If two objects need to communicate Aggregation Whole-part, part can exist separately A o-- B When parts are shared or reusable Composition Strong whole-part, part depends on whole A *-- B When parts are exclusive and lifecycle-bound Inheritance Specialization/ generalization `A < -- B` When \"is-a\"",
      "different ways. Implementation: Use interfaces or abstract base classes with overridden methods. abstract class Shape { +draw(): void } class Circle extends Shape { 3",
      "# Protected: accessible within the class and its subclasses ~ Package: accessible within the same package 2.5. Class Elements Summary Element Description Name Identifier of the class 1"
    ]
  },
  {
    "question": "How do you represent inheritance in PlantUML class diagrams?",
    "response": "In PlantUML, you can represent inheritance (generalization) using the `<|--` symbol. This indicates an \"is-a\" relationship, where the subclass inherits the properties and methods of the superclass.\n\nHere's an example:\n\n```\nclass Animal {\n  +name: String\n  +age: int\n  +speak()\n}\n\nclass Dog <|-- Animal {\n  +breed: String\n  +bark()\n}\n\nclass Cat <|-- Animal {\n  +color: String\n  +meow()\n}\n```\n\nIn this example, the `Dog` and `Cat` classes inherit from the `Animal` class, indicating that a `Dog` and a `Cat` are both types of `Animal`. The subclasses inherit the `name` and `age` properties, as well as the `speak()` method, from the `Animal` class.\n\nThe `<|--` symbol is used to denote the inheritance relationship between the classes. This is the correct way to represent inheritance in PlantUML class diagrams.",
    "chunks": [
      "House *-- Room 3.5. Generalization (Inheritance) Definition: Indicates an “is-a” relationship where a subclass inherits features of the superclass. When to Use: To represent specialization/generalization hierarchy. When Not to Use: Avoid overuse; prefer composition for shared behavior where inheritance doesn’t model a true “is-a” relation. Person <|-- Employee 3.6. Realization (Interface Implementation) Definition: A semantic relationship where a class implements an interface.",
      "UML Class Diagram Design Guide 1. Introduction This document is a comprehensive knowledge base for designing high-quality UML class diagrams from system  descriptions.  It  defines  key  UML  concepts,  class  design  principles,  PlantUML  examples, relationship  types  and  their  correct  usage,  design  patterns,  and  object-oriented  principles  like encapsulation and polymorphism. 2. UML Class Diagram Concepts 2.1. Class",
      "class Scene { -location: String -shootDate: Date } class Actor { -name: String } class Prop { -name: String } class Production { -title: String } Director \"1\" --> \"1..*\" Scene : manages Scene \"1\" --> \"1..*\" Actor : involves Scene \"1\" --> \"0..*\" Prop : uses Actor \"1\" --> \"*\" Scene : scheduled in Production \"1\" *-- \"1..*\" Prop : owns @enduml 5"
    ]
  },
  {
    "question": "What symbols are used for composition and aggregation in PlantUML?",
    "response": "In PlantUML, the symbols used for composition and aggregation are:\n\n1. **Composition**: The \"diamond\" symbol is used to represent composition. The diamond is placed at the end of the association line connected to the \"whole\" class.\n\n   PlantUML syntax: `A *-- B`\n\n2. **Aggregation**: The \"hollow diamond\" symbol is used to represent aggregation. The hollow diamond is placed at the end of the association line connected to the \"whole\" class.\n\n   PlantUML syntax: `A o-- B`\n\nHere's an example of how these symbols would be used in a PlantUML class diagram:\n\n```\n@startuml\nclass Car {\n  - make: String\n  - model: String\n}\n\nclass Wheel {\n  - size: Integer\n}\n\nCar *-- Wheel : has 4\nCar o-- Engine : has 1\n@enduml\n```\n\nIn this example:\n\n- The `Car` class has a composition relationship with the `Wheel` class, indicating that a `Car` object contains 4 `Wheel` objects, and the lifecycle of the `Wheel` objects is dependent on the `Car` object.\n- The `Car` class has an aggregation relationship with the `Engine` class, indicating that a `Car` object has an `Engine` object, but the `Engine` object can exist independently of the `Car` object.",
    "chunks": [
      "9. Summary Table Concept Definition Example UML Syntax When to Use Class Blueprint for objects class Person {} Represent structure and behavior of entities Association Structural link between classes A --> B If two objects need to communicate Aggregation Whole-part, part can exist separately A o-- B When parts are shared or reusable Composition Strong whole-part, part depends on whole A *-- B When parts are exclusive and lifecycle-bound Inheritance Specialization/ generalization `A < -- B` When \"is-a\"",
      "class Scene { -location: String -shootDate: Date } class Actor { -name: String } class Prop { -name: String } class Production { -title: String } Director \"1\" --> \"1..*\" Scene : manages Scene \"1\" --> \"1..*\" Actor : involves Scene \"1\" --> \"0..*\" Prop : uses Actor \"1\" --> \"*\" Scene : scheduled in Production \"1\" *-- \"1..*\" Prop : owns @enduml 5",
      "Booch, G., Rumbaugh, J., & Jacobson, I. (2005). The Unified Modeling Language User Guide. UML 2.5 Specification (OMG) Fowler , M. (2004). UML Distilled: A Brief Guide to the Standard Object Modeling Language. PlantUML Documentation: https://plantuml.com/class-diagram GoF Design Patterns 6"
    ]
  },
  {
    "question": "How do you add attributes to a class in PlantUML?",
    "response": "To add attributes to a class in PlantUML, you can use the following syntax:\n\n```\nclass ClassName {\n    -attribute1: type\n    -attribute2: type\n    +attribute3: type\n    #attribute4: type\n    ~attribute5: type\n}\n```\n\nHere's an explanation of the different access modifiers used for the attributes:\n\n- `-` (private): The attribute is only accessible within the class.\n- `+` (public): The attribute is accessible from anywhere.\n- `#` (protected): The attribute is accessible within the class and its subclasses.\n- `~` (package): The attribute is accessible within the same package.\n\nFor example, in your provided code:\n\n```plantuml\nclass Scene {\n    -location: String\n    -shootDate: Date\n}\n\nclass Actor {\n    -name: String\n}\n\nclass Prop {\n    -name: String\n}\n\nclass Production {\n    -title: String\n}\n```\n\nIn this case, all the attributes are declared as private (`-`) because they are intended to be encapsulated within the class.\n\nRemember that the attributes should be declared before any methods or other class members in the PlantUML syntax.",
    "chunks": [
      "encapsulation and polymorphism. 2. UML Class Diagram Concepts 2.1. Class Academic Definition: A class is a fundamental building block in UML that represents a set of objects sharing  common  structure  (attributes),  behavior  (operations),  relationships,  and  semantics.  (Booch, Rumbaugh, & Jacobson, 2005) class User { -id: int -name: String +login(): boolean } 2.2. Attribute Academic Definition: An attribute is a named property of a class that defines a value range associated",
      "class Scene { -location: String -shootDate: Date } class Actor { -name: String } class Prop { -name: String } class Production { -title: String } Director \"1\" --> \"1..*\" Scene : manages Scene \"1\" --> \"1..*\" Actor : involves Scene \"1\" --> \"0..*\" Prop : uses Actor \"1\" --> \"*\" Scene : scheduled in Production \"1\" *-- \"1..*\" Prop : owns @enduml 5",
      "different ways. Implementation: Use interfaces or abstract base classes with overridden methods. abstract class Shape { +draw(): void } class Circle extends Shape { 3"
    ]
  },
  {
    "question": "What's the syntax for adding methods to a class in PlantUML?",
    "response": "In PlantUML, the syntax for adding methods to a class is as follows:\n\n```\nclass ClassName {\n    [+|-|#|~] methodName(parameters): returnType\n}\n```\n\nHere's a breakdown of the syntax:\n\n- `ClassName`: The name of the class.\n- `+`: Public visibility (default if not specified).\n- `-`: Private visibility.\n- `#`: Protected visibility.\n- `~`: Package-private visibility.\n- `methodName`: The name of the method.\n- `parameters`: The parameters of the method, enclosed in parentheses and separated by commas.\n- `returnType`: The return type of the method.\n\nHere's an example:\n\n```\nclass Person {\n    +getName(): String\n    -setName(name: String): void\n    #getAge(): int\n    ~calculateBMI(height: double, weight: double): double\n}\n```\n\nIn this example, the `Person` class has four methods:\n- `getName()`: a public method that returns a `String`\n- `setName(name: String)`: a private method that takes a `String` parameter and returns `void`\n- `getAge()`: a protected method that returns an `int`\n- `calculateBMI(height: double, weight: double)`: a package-private method that takes two `double` parameters and returns a `double`\n\nRemember that the visibility modifiers (`+`, `-`, `#`, `~`) are optional, and if not specified, the default visibility is public.",
    "chunks": [
      "9. Summary Table Concept Definition Example UML Syntax When to Use Class Blueprint for objects class Person {} Represent structure and behavior of entities Association Structural link between classes A --> B If two objects need to communicate Aggregation Whole-part, part can exist separately A o-- B When parts are shared or reusable Composition Strong whole-part, part depends on whole A *-- B When parts are exclusive and lifecycle-bound Inheritance Specialization/ generalization `A < -- B` When \"is-a\"",
      "different ways. Implementation: Use interfaces or abstract base classes with overridden methods. abstract class Shape { +draw(): void } class Circle extends Shape { 3",
      "# Protected: accessible within the class and its subclasses ~ Package: accessible within the same package 2.5. Class Elements Summary Element Description Name Identifier of the class 1"
    ]
  },
  {
    "question": "How do you specify visibility modifiers in PlantUML?",
    "response": "In PlantUML, you can specify visibility modifiers for class members (attributes and operations) using the following syntax:\n\n- `+` for public visibility\n- `-` for private visibility\n- `#` for protected visibility\n- `~` for package-private (default) visibility\n\nHere's an example:\n\n```plantuml\nclass MyClass {\n    + publicAttribute: String\n    - privateAttribute: Integer\n    # protectedAttribute: Boolean\n    ~ packagePrivateAttribute: Double\n    \n    + publicOperation(): void\n    - privateOperation(): void\n    # protectedOperation(): void\n    ~ packagePrivateOperation(): void\n}\n```\n\nIn this example, the class `MyClass` has four attributes with different visibility modifiers, and four operations with different visibility modifiers.\n\nThe visibility modifiers are placed before the attribute or operation name, and they determine the accessibility of the class members. Public members can be accessed from anywhere, private members can only be accessed within the class, protected members can be accessed within the class and its subclasses, and package-private members can be accessed within the same package.\n\nRemember that the visibility modifiers are an important part of the UML class diagram, as they help to understand the encapsulation and information hiding principles of object-oriented design.",
    "chunks": [
      "with each instance of the class. Attributes model the internal state of objects. (Fowler , 2004) 2.3. Operation (Method) Academic Definition: An operation is a specification of a transformation or computation that an object may be called to execute. It represents behavior associated with a class. (UML 2.5 Specification) 2.4. Visibility Modifiers + Public: visible to all - Private: accessible only within the class # Protected: accessible within the class and its subclasses",
      "Booch, G., Rumbaugh, J., & Jacobson, I. (2005). The Unified Modeling Language User Guide. UML 2.5 Specification (OMG) Fowler , M. (2004). UML Distilled: A Brief Guide to the Standard Object Modeling Language. PlantUML Documentation: https://plantuml.com/class-diagram GoF Design Patterns 6",
      "class Scene { -location: String -shootDate: Date } class Actor { -name: String } class Prop { -name: String } class Production { -title: String } Director \"1\" --> \"1..*\" Scene : manages Scene \"1\" --> \"1..*\" Actor : involves Scene \"1\" --> \"0..*\" Prop : uses Actor \"1\" --> \"*\" Scene : scheduled in Production \"1\" *-- \"1..*\" Prop : owns @enduml 5"
    ]
  },
  {
    "question": "How do you create abstract classes and interfaces in PlantUML?",
    "response": "To create abstract classes and interfaces in PlantUML, you can use the following syntax:\n\n1. **Abstract Class**:\n   - Declare an abstract class using the `abstract class` keyword.\n   - Define abstract methods within the abstract class using the `{abstract}` keyword.\n   - Concrete classes that extend the abstract class must implement the abstract methods.\n\nExample:\n```\nabstract class Shape {\n  {abstract} +draw(): void\n}\n\nclass Circle extends Shape {\n  +draw(): void {\n    ' implementation of draw() method\n  }\n}\n```\n\n2. **Interface**:\n   - Declare an interface using the `interface` keyword.\n   - Define abstract methods within the interface without any implementation.\n   - Classes that implement the interface must provide the implementation for the abstract methods.\n\nExample:\n```\ninterface Drawable {\n  +draw(): void\n}\n\nclass Circle implements Drawable {\n  +draw(): void {\n    ' implementation of draw() method\n  }\n}\n```\n\nIn the context of the provided summary table:\n\n- **Abstract Class**: Use an abstract class when you want to provide a common base implementation for a set of related classes, and you want to enforce the implementation of certain methods. Abstract classes can also have concrete methods and member variables.\n- **Interface**: Use an interface when you want to define a contract or a set of methods that a class must implement, without providing any implementation details. Interfaces are useful for defining a common API or behavior that multiple classes can implement.\n\nThe key differences between abstract classes and interfaces are:\n- Abstract classes can have both abstract and concrete methods, while interfaces can only have abstract methods (except for default and static methods in Java 8+).\n- Abstract classes can have instance variables and constructors, while interfaces cannot have instance variables or constructors.\n- A class can extend only one abstract class, but it can implement multiple interfaces.",
    "chunks": [
      "9. Summary Table Concept Definition Example UML Syntax When to Use Class Blueprint for objects class Person {} Represent structure and behavior of entities Association Structural link between classes A --> B If two objects need to communicate Aggregation Whole-part, part can exist separately A o-- B When parts are shared or reusable Composition Strong whole-part, part depends on whole A *-- B When parts are exclusive and lifecycle-bound Inheritance Specialization/ generalization `A < -- B` When \"is-a\"",
      "different ways. Implementation: Use interfaces or abstract base classes with overridden methods. abstract class Shape { +draw(): void } class Circle extends Shape { 3",
      "Booch, G., Rumbaugh, J., & Jacobson, I. (2005). The Unified Modeling Language User Guide. UML 2.5 Specification (OMG) Fowler , M. (2004). UML Distilled: A Brief Guide to the Standard Object Modeling Language. PlantUML Documentation: https://plantuml.com/class-diagram GoF Design Patterns 6"
    ]
  }
]
    
    """
    # Evaluate batch
    results = evaluator.evaluate_batch(test_cases)
    
    # Generate report
    """
    report = evaluator.generate_report(results)
    print(report)"""
    embedded = test_cases[0]["embedding_model"] 
    split = test_cases[0]["splitter_config"]
    retriever = test_cases[0]["retriever"]
    path = "EvaluationRAGRESULTS/evaluation_results_" + embedded + "_" + split + "_" + retriever + ".xlsx"
    # Export to Excel
    evaluator.export_evaluation_to_excel(results, path)
    
    # Print details for each EvaluationResult
    print("\nIndividual Evaluation Details:")
    for i, result in enumerate(results["individual_results"], 1):
        print(f"\nResult {i}:")
        print(f"  Accuracy: {result.accuracy:.3f}")
        print(f"  Semantic Similarity: {result.semantic_similarity:.3f}")
        print(f"  Completeness: {result.completeness:.3f}")
        print(f"  Relevance: {result.relevance:.3f}")
        print(f"  Syntax Correctness: {result.syntax_correctness:.3f}")
        print(f"  Overall Score: {result.overall_score:.3f}")
        print(f"  Details: {json.dumps(result.details, indent=2)}")

    return results


if __name__ == "__main__":
    run_comprehensive_test()
    """ 
    documents = load_documents()
    embedding_models = get_embedding_models()
    text_splitter_configs = get_text_splitter_configs()

    embedding_model = embedding_models["all-MiniLM-L6-v2"]  # Example model
    split_config = text_splitter_configs["default"]  # Example config
    vector_store, num_chunks = create_vector_store(documents, embedding_model, split_config)
                
                
    retrievers = create_retrievers(vector_store, documents)
    retriever = retrievers["bm25_k5"]
    rag = RAGSystemConfig(retriever= retriever, vector_store=vector_store, question="How do you start and end a PlantUML class diagram?")

    result = []
    response, context, retrieval_time = rag.generate_results()
    result.append({
                        "question": "How do you start and end a PlantUML class diagram?",
                        "response": response.content,
                        "chunks": context,
                        "embedding_model": "all-MiniLM-L6-v2",
                        "splitter_config": "default",
                        "retriever": "bm25_k5",
                        "retrieval_time": retrieval_time,
                        "num_chunks": num_chunks
                    })
    print(result)
    example_usage(result)
    """
