#https://medium.com/@rajveer.rathod1301/evaluating-llm-responses-with-deepeval-library-a-comprehensive-practical-guide-e55ef1f9eeab

from langchain_ollama.llms import OllamaLLM
from langchain_anthropic import ChatAnthropic
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import json
import re
from typing import Dict, List, Optional
import os
import asyncio
import logging
from dotenv import load_dotenv
class TestOllamaLLM(DeepEvalBaseLLM):
    """Class to implement Ollama for DeepEval"""
    
    def __init__(self, model_name: str):
        self.model = OllamaLLM(model=model_name)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.predict(prompt)

    async def a_generate(self, prompt: str) -> str:
        return await self.model.apredict(prompt)

    def get_model_name(self):
        return f"Ollama - {self.model.model}"

class TestAnthropicLLM(DeepEvalBaseLLM):
    """Class to implement Anthropic (Claude) for DeepEval via LangChain."""

    def __init__(self, model_name: str, api_key: Optional[str] = None):
        kwargs = {"model": model_name}
        if api_key:
            kwargs["api_key"] = api_key
        self.model = ChatAnthropic(**kwargs)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.predict(prompt)

    async def a_generate(self, prompt: str) -> str:
        return await self.model.apredict(prompt)

    def get_model_name(self):
        return f"Anthropic - {self.model.model}"

class ClassDiagramMetric(BaseMetric):
    """Metric for evaluating class diagram generation based on syntactic, semantic, and pragmatic quality"""
    def __init__(self, threshold: float = 0.7, llm_judge_model: Optional[str] = None, llm_judge_provider: Optional[str] = None):
        self.threshold = threshold
        self.scorer = Scorer()
        # Initialize LLM judge if specified (supports Ollama and Anthropic)
        self.llm_judge = None
        if llm_judge_model:
            provider = (llm_judge_provider or os.getenv("EVAL_JUDGE_PROVIDER") or "ollama").lower()
            if provider.startswith("anth") or ("claude" in llm_judge_model.lower()):
                self.llm_judge = ChatAnthropic(model=llm_judge_model)
            else:
                self.llm_judge = OllamaLLM(model=llm_judge_model)
        # UML syntax patterns
        self.syntax_patterns = {
            'class_def': r'class\s+\w+\s*{',
            'attribute_def': r'[+-]\s*\w+\s*:\s*\w+',
            'method_def': r'[+-]\s*\w+\s*\([^)]*\)\s*(:\s*\w+)?',
            'relationship': r'(\w+)\s*(--|..|<|>|-\*|\*-|o-|-o)\s*(\w+)',
            'multiplicity': r'"[0-9*.]+"'
        }
        
    def _extract_classes(self, puml_code: str) -> List[str]:
        """Extract class names from PlantUML code"""
        class_pattern = r'class\s+(\w+)'
        return re.findall(class_pattern, puml_code)
    
    def _extract_relationships(self, puml_code: str) -> List[str]:
        """Extract relationships from PlantUML code"""
        rel_pattern = r'(\w+)\s*(--|..|<|>|-\*|\*-|o-|-o)\s*(\w+)'
        return re.findall(rel_pattern, puml_code)

    def _llm_judge_evaluation(self, actual: str, expected: str, input) -> float:
        """Use LLM to judge the quality of the generated diagram"""
        if not self.llm_judge:
            return 1.0  # Skip LLM judgment if no judge model specified
            
        prompt = f"""
                You are an expert UML evaluator and software engineering reviewer. Your task is to assess the quality of a UML class diagram based on system requirements and a benchmark ("golden") diagram. Evaluate the diagram across the following dimensions:

                1. **UML Completeness** – Are all relevant elements present (classes, attributes, methods, relationships)?
                2. **UML Quality** – Are design principles respected? Are names, types, and relationships coherent and correctly applied?
                3. **Requirement Coverage** – Does the class diagram faithfully implement the described system requirements?
                4. **Clarity & Pragmatics** – Is the diagram understandable and clear for technical stakeholders?

                ---


                - "system_description" -  A description of the system to be modeled:
                    {input}
                - "Class Diagram Generated" - The PlantUML code of the generated class diagram: 
                    {actual}
                - "Benchmark" - The benchmark/golden PlantUML diagram to compare against: 
                    {expected}

                ---

                Return only a JSON object with the following structure:

                
                    
                    "uml_completeness_score": <float from 0.0 to 1.0>,
                    "uml_quality_score": <float from 0.0 to 1.0>,
                    "requirement_coverage_score": <float from 0.0 to 1.0>,
                    "pragmatic_clarity_score": <float from 0.0 to 1.0>,
                    "overall_score": <float from 0.0 to 1.0>,
                    "issues":["<brief description of issues found, e.g., missing associations, naming inconsistency, weak abstraction>"],
                    "recommendations": ["<actionable suggestions to improve the diagram>"]
                The output should start with ```json and end with ```
            """
        
        try:
            out = self.llm_judge.invoke(prompt)
            raw_val = getattr(out, "content", out)
            raw = (raw_val if isinstance(raw_val, str) else str(raw_val)).strip()
            logging.info("LLM judge Raw response:\n%s\n", raw)
            m = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
            
            json_str = m.group(1) if m else raw
            data = json.loads(json_str)
            return data["overall_score"] if "overall_score" in data else 0.0
        except Exception as e:
            print(f"LLM judge error: {e}")
            logging.error("LLM judge error: %s", e, exc_info=True)
            return 0.0  # Default score if LLM judgment fails

    def evaluate_diagram_syntax(self,plantuml_content: str) -> float:
        """
        Evaluates the syntactic quality of a PlantUML class diagram.

        Args:
            plantuml_content (str): The PlantUML source code as a string

        Returns:
            dict: A comprehensive evaluation report
        """
        lines = plantuml_content.strip().splitlines()
        errors, warnings, suggestions = [], [], []
        compliance_metrics = {
            "structure_compliance": 1.0,
            "relationship_syntax": 1.0,
            "naming_conventions": 1.0,
            "uml_standards": 1.0
        }
        valid_class_names = set()

        # Regex patterns
        class_decl = re.compile(r'^(abstract\s+)?class\s+(\w+)', re.IGNORECASE)
        interface_decl = re.compile(r'^interface\s+(\w+)', re.IGNORECASE)
        attr_pattern = re.compile(r'^[ \t]*[+#\-~]\s*\w+\s*:\s*\w+')
        method_pattern = re.compile(r'^[ \t]*[+#\-~]\s*\w+\s*\([^)]*\)\s*(?::\s*\w+)?')
        rel_pattern = re.compile(r'(\w+)(?:\s+"?[\d.*]+"?)?\s+([*o<]?-{1,2}[*o>"]?)\s+(?:"?[\d.*]+"?\s+)?(\w+)(\s*:\s*(.*))?')
        multiplicity_pattern = re.compile(r'"?[\d.*]+"?')

        if "@startuml" not in plantuml_content or "@enduml" not in plantuml_content:
            errors.append({
                "type": "syntax_error",
                "line": 0,
                "message": "Missing @startuml or @enduml",
                "severity": "critical"
            })
            compliance_metrics["structure_compliance"] = 0.0

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("'"):  # skip comments and blanks
                continue

            # Class declarations
            if class_decl.match(stripped):
                class_name = class_decl.match(stripped).group(2)
                valid_class_names.add(class_name)
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                    warnings.append({
                        "type": "naming_convention",
                        "line": i,
                        "message": f"Class name '{class_name}' should follow PascalCase convention",
                        "severity": "minor"
                    })
                    compliance_metrics["naming_conventions"] -= 0.1

            elif interface_decl.match(stripped):
                interface_name = interface_decl.match(stripped).group(1)
                valid_class_names.add(interface_name)

            elif attr_pattern.match(stripped):
                if ':' not in stripped:
                    errors.append({
                        "type": "syntax_error",
                        "line": i,
                        "message": "Attribute declaration missing type annotation",
                        "severity": "critical"
                    })
            elif method_pattern.match(stripped):
                if not stripped.endswith(")") and ':' not in stripped:
                    warnings.append({
                        "type": "style_warning",
                        "line": i,
                        "message": "Method declaration may be missing return type",
                        "severity": "minor"
                    })

            elif any(op in stripped for op in ['--','<|--','--|>','<--','--*','--o', '*--', 'o--', '-->']):
                rel_match = rel_pattern.match(stripped)
                if not rel_match:
                    errors.append({
                        "type": "syntax_error",
                        "line": i,
                        "message": f"Invalid relationship syntax: '{stripped}'",
                        "severity": "critical"
                    })
                    compliance_metrics["relationship_syntax"] -= 0.2
                else:
                    left_class, arrow, right_class, _, label = rel_match.groups()
                    if left_class not in valid_class_names or right_class not in valid_class_names:
                        warnings.append({
                            "type": "invalid_reference",
                            "line": i,
                            "message": f"Relationship refers to undeclared class(es): {left_class}, {right_class}",
                            "severity": "minor"
                        })
                        compliance_metrics["uml_standards"] -= 0.1
                    if not multiplicity_pattern.search(stripped):
                        warnings.append({
                            "type": "missing_cardinality",
                            "line": i,
                            "message": f"Association between {left_class} and {right_class} missing multiplicity",
                            "severity": "minor"
                        })
                        compliance_metrics["uml_standards"] -= 0.05
                    if label and not re.match(r'^[a-z][a-zA-Z0-9\s]*$', label.strip()):
                        warnings.append({
                            "type": "naming_convention",
                            "line": i,
                            "message": f"Association label '{label.strip()}' should be a verb phrase starting lowercase",
                            "severity": "minor"
                        })
                        compliance_metrics["naming_conventions"] -= 0.05

        for k in compliance_metrics:
            compliance_metrics[k] = max(0.0, round(compliance_metrics[k], 2))

        overall_score = round(
            0.4 * compliance_metrics["structure_compliance"] +
            0.3 * compliance_metrics["relationship_syntax"] +
            0.2 * compliance_metrics["naming_conventions"] +
            0.1 * compliance_metrics["uml_standards"], 2
        )

        if compliance_metrics["naming_conventions"] < 1.0:
            suggestions.append("Use PascalCase for class names and lowercase verb phrases for relationship labels.")
        if compliance_metrics["relationship_syntax"] < 1.0:
            suggestions.append("Ensure all relationship operators and endpoints follow UML syntax.")
        if compliance_metrics["uml_standards"] < 1.0:
            suggestions.append("Add multiplicity to associations and verify class references are valid.")

        data = {
            "overall_score": overall_score,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "compliance_metrics": compliance_metrics
        }
        logging.info("Syntactic evaluation data: %s", json.dumps(data, indent=2))
        return data["overall_score"]
    def evaluate_diagram_pragmatic(self, actual: str, expected: str) -> float:
        prompt = f""" 
        You are an expert UML evaluator. Evaluate the pragmatic quality of this PlantUML class diagram from 0.0 to 1.0, focusing on understandability and clarity for stakeholders.  
        Output **only** a JSON object with these keys:

        - "score": <float between 0.0 and 1.0>  
        - "issues": <list of strings>, describing any pragmatic problems (e.g. redundant attributes, weak subclass distinction, inconsistent styling)  
        - "recommendations": <list of strings>, actionable suggestions to improve clarity

        The output should start with ```json and end with ```

        Class Diagram:
        {actual}

        Compare against expected best practices:
        {expected}

          
        """
        try:
            raw = self.llm_judge.predict(prompt).strip()
           
            m = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
            logging.info("Pragmatic Raw response:\n%s\n", raw)

            json_str = m.group(1) if m else raw
            data = json.loads(json_str)

            # clamp score
            data['score'] = min(max(data.get('score', 0.0), 0.0), 1.0)
         
            return data['score']
        except Exception as e:
            print(f"Pragmatic quality evaluation error: {e}")
            logging.error("LLM judge error: %s", e, exc_info=True)
            # fallback
            data = {
                "score": 0.5,
                "issues": [],
                "recommendations": []
            }
            return data['score']
    def evaluate_diagram_semantic(self, actual: str, expected: str) -> float:
        prompt = f"""
            You are an expert UML semantic evaluator. Evaluate the semantic quality of this PlantUML class diagram in terms of **accuracy** (validity) and **coverage** (completeness) for the intended domain.  

            Output **only** a JSON object with these keys:  
            - `"validity_score"`: float between 0.0 and 1.0, where 1.0 means every element and relationship accurately reflects the domain.  
            - `"completeness_score"`: float between 0.0 and 1.0, where 1.0 means all necessary elements and relationships are present.  
            - `"issues"`: list of strings, each describing a specific semantic error found.  
            - `"recommendations"`: list of strings, each suggesting how to correct or improve the diagram.

            The output should start with ```json and end with ```
            **Evaluate on**:  
            1. **Validity** – do all classes, attributes, operations, and relationships faithfully represent domain concepts?  
            2. **Completeness** – are any classes, attributes, operations, or relationships missing or omitted?  

            **Watch for these common semantic errors**:  
            - Incorrect cardinality or multiplicity specifications  
            - Use of aggregation where a simple association is required  
            - Attributes placed in the wrong class  
            - Operations defined where they cannot be realized with existing data  
            - Operations that cannot be implemented using the current attributes/relationships  

            **PlantUML Diagram:**  
            {actual}

            **Benchmark Diagram:**  
            {expected}

        """
        try:
            raw = self.llm_judge.predict(prompt).strip()
        
            m = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
            json_str = m.group(1) if m else raw
            logging.info("Semantic Raw response:\n%s\n", raw)
            data = json.loads(json_str)
         
            score =  (data['validity_score'] + data['completeness_score']) / 2
            return score
        except Exception as e:
            print(f"Semantic quality evaluation error: {e}")
            logging.error("Semantic quality evaluation error: %s", e, exc_info=True)
            # fallback
            data = {
                "score": 0.5,
                "issues": [],
                "recommendations": []
            }
            return data['score']

    def measure(self, test_case: LLMTestCase):
        actual = test_case.actual_output
        expected = test_case.expected_output
        input = test_case.input
        # Calculate quality scores
        syntactic_score = self.evaluate_diagram_syntax(actual)
        semantic_scores = self.evaluate_diagram_semantic(actual, expected)
        pragmatic_score = self.evaluate_diagram_pragmatic(actual, expected)
        llm_judge_score = self._llm_judge_evaluation(actual, expected, input)
        
        # Store individual scores
        self.detailed_scores = {
            "syntactic_quality": syntactic_score,
            "semantic_quality": semantic_scores,
            "pragmatic_quality":pragmatic_score,
            "llm_judge": llm_judge_score
        }
        
        # Calculate final score (weighted average)
        weights = {
            "syntactic_quality": 0.25,
            "semantic_quality": 0.45,
            "pragmatic_quality": 0.15,
            "llm_judge": 0.20
        }
        
        # Print individual scores
        print("\nDetailed Scores:")
        for metric, score in self.detailed_scores.items():
            print(f"  {metric}: {score:.3f}")
            
        self.score = sum(self.detailed_scores[k] * weights[k] for k in weights.keys())
        print(f"  Average Score: {self.score:.3f}")
        
        self.success = self.score >= self.threshold
        return self.score

    def _evaluate_structure(self, actual: str, expected: str) -> float:
        """Evaluate structural similarity of class diagrams"""
        actual_classes = set(self._extract_classes(actual))
        expected_classes = set(self._extract_classes(expected))
        
        actual_rels = set(map(str, self._extract_relationships(actual)))
        expected_rels = set(map(str, self._extract_relationships(expected)))
        
        class_similarity = len(actual_classes & expected_classes) / max(len(expected_classes), 1)
        rel_similarity = len(actual_rels & expected_rels) / max(len(expected_rels), 1)
        
        return (class_similarity + rel_similarity) / 2

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Class Diagram Metric"

class RougeMetric(BaseMetric):
    """Rouge metric for text similarity evaluation"""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type="rouge1"  # Using ROUGE-1 for evaluation
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"

def setup_evaluation(
    test_data_path: str,
    model_names: Optional [List[str]] = None,
    provider: Optional[str] = None,
    llm_judge_model: Optional[str] = None,
    llm_judge_provider: Optional[str] = None,
):
    """Setup the evaluation environment"""
    # Choose provider (env override: EVAL_PROVIDER)
    provider = (provider or os.getenv("EVAL_PROVIDER") or "ollama").lower()

    # Initialize models based on provider
    if provider.startswith("anth"):
        models = [TestAnthropicLLM(name, api_key=os.getenv("ANTHROPIC_API_KEY")) for name in model_names]
    else:
        models = [TestOllamaLLM(name) for name in model_names]
    
    # Load test cases
    dataset = EvaluationDataset()
    dataset.add_test_cases_from_json_file(
        file_path=test_data_path,
        input_key_name="input",
        actual_output_key_name="actual_output",
        expected_output_key_name="expected_output",
        
    )
    
    # Initialize metrics (judge configurable via args/env)
    judge_model = llm_judge_model or os.getenv("EVAL_JUDGE_MODEL") or "llama3.1:8b"
    judge_provider = (llm_judge_provider or os.getenv("EVAL_JUDGE_PROVIDER") or "ollama").lower()
    print(f"Using LLM judge model: {judge_model} from provider: {judge_provider}")
    metrics = [
        ClassDiagramMetric(threshold=0.7, llm_judge_model=judge_model, llm_judge_provider=judge_provider),
        RougeMetric(threshold=0.5),
    ]
    
    return models, dataset, metrics

async def evaluate_models():
    """Main evaluation function"""
    logging.basicConfig(
        filename="evaluation_results.log",   # where to write
        filemode="w",                        # overwrite each run ("a" to append)
        level=logging.INFO,                  # capture INFO and above
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    models, dataset, metrics = setup_evaluation(
        provider="anthropic", 
        llm_judge_provider="anthropic", 
        model_names=["dummy"],
        llm_judge_model="claude-3-haiku-20240307",
        test_data_path="test_data.json"
    )
    
    results = {}
    
    for model in models:
        model_results = {
            "metrics": {},
            "examples": []
        }
        logging.info("Starting evaluation of test cases...")
        
        print("\n" + "="*50)
        print("Starting evaluation of test cases...")
        print("="*50)
        
        for i, test_case in enumerate(dataset, 1):
            print(f"\nTest Case #{i}")
            logging.info("Evaluating test case #%d", i)
            print("-" * 20)
            
            example_results = {}
            for metric in metrics:
                print(f"\nApplying {metric.__name__}...")
                score = await metric.a_measure(test_case)
                
                if metric.__name__ not in model_results["metrics"]:
                    model_results["metrics"][metric.__name__] = []
                model_results["metrics"][metric.__name__].append(score)
                
                example_results[metric.__name__] = score
                print(f"Score for {metric.__name__}: {score:.3f}")
            
            model_results["examples"].append({
                "input": test_case.input,
                "actual_output": test_case.actual_output,
                "expected": test_case.expected_output,
                "scores": example_results
            })
            
            print("\nScores for this test case:")
            logging.info("Scores for test case #%d", i)
            for metric_name, score in example_results.items():
                print(f"  {metric_name}: {score:.3f}")
                logging.info("Score for %s: %.3f", metric_name, score)
            print("-" * 50)
        
        results["Evaluation Results"] = model_results
    
    return results

# Modified main section
if __name__ == "__main__":
  
    
    async def main():
        load_dotenv()
        results = await evaluate_models()
        # We don't need to print the averages anymore since we're showing individual scores
        
    asyncio.run(main())
