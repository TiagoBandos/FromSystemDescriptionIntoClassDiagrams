from typing import List, Tuple,Dict, Set
from difflib import SequenceMatcher
import re
from langchain_ollama.llms import OllamaLLM
import json
def extract_classes(puml_code: str) -> List[str]:
    """Extract class names from PlantUML code"""
    class_pattern = r'class\s+(\w+)'
    return re.findall(class_pattern, puml_code)

def extract_relationships(puml_code: str) -> List[Tuple[str, str, str]]:
    """Extract relationships from PlantUML code"""
    rel_pattern = r'''
        (?P<left_class>\w+)
        (?:\s+"(?P<left_multi>[\d.*]+)")?
        \s+(?P<arrow>["*o<]?-{1,2}["*o>"]?)\s+
        (?:"(?P<right_multi>[\d.*]+)"\s+)?     
        (?P<right_class>\w+)
    '''
    return [match.groupdict() for match in re.finditer(rel_pattern, puml_code, re.VERBOSE)]

def extract_methods(puml_code: str) -> Dict[str, List[str]]:
    """
    Extracts methods from each class in PlantUML code.
    Returns a dictionary: {class_name: [method1(), method2()]}
    """
    class_blocks = re.findall(r'class\s+(\w+)\s*\{([^}]*)\}', puml_code, re.DOTALL)
    methods_by_class = {}

    for class_name, body in class_blocks:
        methods = re.findall(r'[+#\-~]?\s*(\w+)\s*\([^)]*\)\s*:\s*\w+', body)
        methods_by_class[class_name] = methods

    return methods_by_class

def extract_attributes(puml_code: str) -> Dict[str, List[str]]:
    """
    Extracts attributes from each class in PlantUML code.
    Returns a dictionary: {class_name: [attribute1, attribute2]}
    """
    class_blocks = re.findall(r'class\s+(\w+)\s*\{([^}]*)\}', puml_code, re.DOTALL)
    attributes_by_class = {}

    for class_name, body in class_blocks:
        attributes = re.findall(r'[+#\-~]?\s*(\w+)\s*:\s*\w+', body)
        attributes_by_class[class_name] = attributes

    return attributes_by_class


def evaluate_plantuml_syntax(plantuml_content: str) -> dict:
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

        elif any(op in stripped for op in ['--', '<|', '*--', 'o--', '-->']):
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

    return {
        "overall_score": overall_score,
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "compliance_metrics": compliance_metrics
    }


def evaluate_semantic_quality(generated_puml: str, benchmark_puml: str) -> Dict:
    """
    Compares two PlantUML class diagrams to evaluate semantic quality.
    
    Args:
        generated_puml (str): Generated class diagram (PlantUML code)
        benchmark_puml (str): Ground truth class diagram (PlantUML code)
        
    Returns:
        dict: Evaluation report with completeness and validity details
    """
    # Extract components
    gen_classes = set(extract_classes(generated_puml))
    bench_classes = set(extract_classes(benchmark_puml))

    gen_attrs = extract_attributes(generated_puml)
    bench_attrs = extract_attributes(benchmark_puml)

    gen_methods = extract_methods(generated_puml)
    bench_methods = extract_methods(benchmark_puml)

    gen_rels = extract_relationships(generated_puml)
    bench_rels = extract_relationships(benchmark_puml)

    def rel_key(r):  # Normalize for comparison
        return (r['left_class'], r['arrow'], r['right_class'])

    gen_rel_set = set(map(rel_key, gen_rels))
    bench_rel_set = set(map(rel_key, bench_rels))

    # Compare classes
    missing_classes = list(bench_classes - gen_classes)
    extra_classes = list(gen_classes - bench_classes)

    # Compare attributes/methods per class
    missing_attrs, extra_attrs = {}, {}
    missing_methods, extra_methods = {}, {}

    for cls in bench_classes:
        if cls in bench_attrs:
            missing = set(bench_attrs.get(cls, [])) - set(gen_attrs.get(cls, []))
            if missing:
                missing_attrs[cls] = list(missing)

        if cls in bench_methods:
            missing = set(bench_methods.get(cls, [])) - set(gen_methods.get(cls, []))
            if missing:
                missing_methods[cls] = list(missing)

    for cls in gen_classes:
        if cls in gen_attrs:
            extra = set(gen_attrs.get(cls, [])) - set(bench_attrs.get(cls, []))
            if extra:
                extra_attrs[cls] = list(extra)

        if cls in gen_methods:
            extra = set(gen_methods.get(cls, [])) - set(bench_methods.get(cls, []))
            if extra:
                extra_methods[cls] = list(extra)

    # Compare relationships
    missing_rels = list(bench_rel_set - gen_rel_set)
    extra_rels = list(gen_rel_set - bench_rel_set)

    # Compute completeness/validity
    total_classes = len(bench_classes)
    total_rels = len(bench_rel_set)
    class_score = 1 - len(missing_classes) / total_classes if total_classes else 1.0
    rel_score = 1 - len(missing_rels) / total_rels if total_rels else 1.0
    attr_score = 1 - sum(len(v) for v in missing_attrs.values()) / sum(len(v) for v in bench_attrs.values()) if bench_attrs else 1.0
    method_score = 1 - sum(len(v) for v in missing_methods.values()) / sum(len(v) for v in bench_methods.values()) if bench_methods else 1.0

    semantic_score = round(0.3 * class_score + 0.3 * rel_score + 0.2 * attr_score + 0.2 * method_score, 2)

    return {
        "semantic_score": semantic_score,
        "class_completeness": round(class_score, 2),
        "relationship_completeness": round(rel_score, 2),
        "attribute_completeness": round(attr_score, 2),
        "method_completeness": round(method_score, 2),
        "missing_classes": missing_classes,
        "extra_classes": extra_classes,
        "missing_attributes": missing_attrs,
        "extra_attributes": extra_attrs,
        "missing_methods": missing_methods,
        "extra_methods": extra_methods,
        "missing_relationships": missing_rels,
        "extra_relationships": extra_rels
    }





classDiagram = """
@startuml
' Define classes and their attributes/methods
abstract class WorkProduct {
  +description: String
  +percentageCompleted: int
  +validate(): void
}

class ProjectManager {
  +initiateProject(): void
  +terminateProject(): void
}

class Project {
  +startDate: Date
  +endDate: Date
  +status: String
}

class Team {
  +executeProject(): void
}

class Requirements extends WorkProduct {
  +publish(): void
}

class System extends WorkProduct {
  +deploy(): void
}

abstract class Validation {
  +validate(): void
}

class UserValidation extends Validation {
  +validateWithUsers(): void
}

class SystemValidation extends Validation {
  +testAgainstRequirements(): void
}

class PublicationMedia {
  +mediaType: String
}

class Platform {
  +platformType: String
}

' Define relationships
ProjectManager --> Project : manages
Project o-- Team : has
Project "1" *-- "1" Requirements : has input
Project "1" *-- "1" System : has output
WorkProduct "1" --> "1" Validation : validated by
Requirements --> UserValidation : validated by
System --> SystemValidation : validated by
Requirements --> PublicationMedia : published via
System --> Platform : deployed on
@enduml
"""
benchmarkDiagram = """@startuml
class WorkProduct{
    PercentComplete
    Description
    Validate()
}
class System{
    Plataform
    Validate()
    Deploy()
}
class Requirement{
    Media
    Validate()
    Publish()
}
class Project{
    Name
    StartDate
    EndDate
}
class Manager{
    Name
    PhoneNumber
    InitiateProject()
    TerminateProject()
}
class Team{
    Description
}
WorkProduct <|--- System
WorkProduct <|--- Requirement
Requirement -- Project: Input <
System -- Project: Output <
Manager -- Project: Manage >
Team -- Project: Execute >
Manager -- Team : Lead


@enduml
"""

#print(f"Extracted Classes: {extract_classes(classDiagram)}")
#print(f"Extracted Relationships: {extract_relationships(classDiagram)}")
#print  (f"Extracted Methods: {extract_methods(classDiagram)}")
#print  (f"Extracted Attributes: {extract_attributes(classDiagram)}")
#print  (f"Evaluation Result: {evaluate_plantuml_syntax(classDiagram)}")

llm_judge_model="llama3.1:8b"
llm_judge = OllamaLLM(model=llm_judge_model) if llm_judge_model else None

def evaluate_diagram_prag(actual: str, expected: str):
    prompt = f"""
    You are an expert UML evaluator. Evaluate the pragmatic quality of this PlantUML class diagram from 0.0 to 1.0, focusing on understandability and clarity for stakeholders.  
    Output **only** a JSON object with these keys:

    - "score": <float between 0.0 and 1.0>  
    - "issues": <list of strings>, describing any pragmatic problems (e.g. redundant attributes, weak subclass distinction, inconsistent styling)  
    - "recommendations": <list of strings>, actionable suggestions to improve clarity

    Class Diagram:
    {actual}

    Compare against expected best practices:
    {expected}
    """
    try:
        raw = llm_judge.invoke(prompt).strip()
        print(f"Pragmatic quality evaluation raw output: {raw}")
        m = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
     
        json_str = m.group(1) if m else raw
        data = json.loads(json_str)

        # clamp score
        data['score'] = min(max(data.get('score', 0.0), 0.0), 1.0)
        return data
    except Exception as e:
        print(f"Pragmatic quality evaluation error: {e}")
        # fallback
        return {
            "score": 0.5,
            "issues": [],
            "recommendations": []
        }

def evaluate_diagram_semantic(actual: str, expected: str):
    prompt = f"""
        You are an expert UML semantic evaluator. Evaluate the semantic quality of this PlantUML class diagram in terms of **accuracy** (validity) and **coverage** (completeness) for the intended domain.  

        Output **only** a JSON object with these keys:  
        - `"validity_score"`: float between 0.0 and 1.0, where 1.0 means every element and relationship accurately reflects the domain.  
        - `"completeness_score"`: float between 0.0 and 1.0, where 1.0 means all necessary elements and relationships are present.  
        - `"issues"`: list of strings, each describing a specific semantic error found.  
        - `"recommendations"`: list of strings, each suggesting how to correct or improve the diagram.

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
        raw = llm_judge.invoke(prompt).strip()
     
        m = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
        json_str = m.group(1) if m else raw
        data = json.loads(json_str)

        # clamp score
        data['score'] = min(max(data.get('score', 0.0), 0.0), 1.0)
        return data
    except Exception as e:
        print(f"Semantic quality evaluation error: {e}")
        # fallback
        return {
            "score": 0.5,
            "issues": [],
            "recommendations": []
        }

def evaluate_diagram_syntactic(actual: str, expected: str):
    prompt = f"""
        You are an expert UML syntactic evaluator. Evaluate the syntactic quality of this PlantUML class diagram in terms of **structural correctness** (following UML syntax rules) and **notation accuracy** (proper use of UML symbols).  

        Output **only** a JSON object with these keys:  
        - `"syntactic_score"`: float between 0.0 and 1.0, where 1.0 means the diagram is fully compliant with UML syntax and notation.  
        - `"issues"`: list of strings, each describing a specific syntactic error or deviation.  
        - `"recommendations"`: list of strings, each suggesting how to correct or improve the syntax or notation.

        **Evaluate on**:  
        1. **Structural Correctness** – are all classes, associations, inheritances, and multiplicities expressed with correct UML syntax?  
        2. **Notation Accuracy** – are UML symbols (aggregation, composition, associations) used appropriately and consistently?

        **Watch for these common syntactic errors**:  
        - Missing cardinality details for associations (e.g., no multiplicity on an association line)  
        - Inappropriate naming of classes or associations that violates UML naming conventions  
        - Incorrect use of UML symbols (e.g., using `*--` instead of `o--` for aggregation)  

        **PlantUML Diagram:**  
        {actual}

        **Benchmark Diagram :**  
        {expected}
    """
    try:
        raw = llm_judge.invoke(prompt).strip()
     
        m = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
        json_str = m.group(1) if m else raw
        data = json.loads(json_str)

        # clamp score
        data['score'] = min(max(data.get('score', 0.0), 0.0), 1.0)
        return data
    except Exception as e:
        print(f"Semantic quality evaluation error: {e}")
        # fallback
        return {
            "score": 0.5,
            "issues": [],
            "recommendations": []
        }

print(f"Pragmatic quality score: {evaluate_diagram_prag(classDiagram, benchmarkDiagram)}")
print(f"Semantic quality score: {evaluate_diagram_semantic(classDiagram, benchmarkDiagram)}")
print(f"Syntactic quality score: {evaluate_diagram_syntactic(classDiagram, benchmarkDiagram)}")