import json
import logging
import os
import re
import subprocess
import sys
from typing import Any
import uuid
import anthropic
import chromadb
from deepeval import List
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS
import glob

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
# Load environment variables from a .env file
load_dotenv()

# Set the AnthropicAPI_key key environment variable 
anthropic_api_key = os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')

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
    pdf_loader = PyPDFLoader("data/System_Understanding_Project_Management.pdf")
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


def create_vector_store( documents: List[Document], embedding_model, 
                        text_splitter,store_type: str = "chroma") -> Any:
        """Create vector store with given configuration"""
        
        # Create text splitter
    
        
        # Split documents
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        if store_type == "chroma":
            client = chromadb.Client()                   # guarantees blank slate
            collection_name = "ClassDiagramGeneration"

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
        
        return vector_store
        
class RAG :
        def __init__(self):
            self.vector_store = None
            self.retriever = None
            self.clientContext = ChatAnthropic(
                model="claude-3-haiku-20240307",
                temperature=0.2,
                max_tokens=1000
            )
            self.client = ChatAnthropic(
                model="claude-3-7-sonnet-20250219",
                temperature=0.5,
                max_tokens=1000
            )
            self.loaded = False  # Track if already loaded
        def load(self):
            if self.loaded:
                logging.info("Vector store already initialized.")
                return

            logging.info("📄 Loading documents...")
            documents = load_documents()

            logging.info("🧠 Loading embedding model...")
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

            split_config = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=20,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            logging.info("📚 Creating vector store...")
            self.vector_store = create_vector_store(
                documents=documents,
                embedding_model=embedding_model,
                text_splitter=split_config
            )

            self.retriever = self.vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": 3, "fetch_k": max(8, 10)}
            )

            self.loaded = True
            logging.info("✅ RAG vector store initialized.")   
        def generate_results(self):

            questions =  [
               "What is the PlantUML syntax for drawing class diagrams?",
               "What are the best practices for UML class diagrams?",
               "What are the common mistakes in UML class diagrams?",
                "What are the key principles of UML class diagrams?",
            ]
            context_generated = " "
            for question in questions :
                docs = self.retriever.get_relevant_documents(question)
                

                chunks = [doc.page_content for doc in docs]
                context = [self.clean_text(chunk) for chunk in chunks]

                system_prompt = """ You are an expert in PlantUML and UML class diagrams. You have a deep understanding of how to create and interpret UML class diagrams using PlantUML syntax. Your responses should be concise, accurate, and focused on providing clear guidance for creating class diagrams. """
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"**QUESTION** {question}\n\n**CONTEXT**\n" + "\n\n".join(context)),
                ]

                response = self.clientContext.invoke(messages)
                context_generated += "\n" + response.content

            return context_generated    
        def retrieve_context(self, system_description):
            system_prompt = "You are an expert in modelling software engineering.Your responses should be concise, accurate, and focused on providing clear guidance for creating class diagrams. "

           

            docs = self.retriever.get_relevant_documents(system_description)
            chunks = [doc.page_content for doc in docs]
            context = [self.clean_text(chunk) for chunk in chunks]

            user_prompt = f"##TASK Give clear guidance and information need to create a PlantUML code class diagram based on the following system description and external context \n ##CONTEXT {context} \n ##SYSTEM DESCRIPTION {system_description}"
            
            messages = [
                SystemMessage(content = system_prompt),
                HumanMessage(content = user_prompt)
            ]

            context_generated = self.clientContext.invoke(messages)
        
            return context_generated.content
        def clean_text( self,text: str) -> str:
            """Clean and normalize text content"""
            import re
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove special characters that might interfere
            text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\[\]\{\}]', '', text)
            return text.strip()
        def _extract_plantuml_code(self, response_text: str) -> str:
                """Extract PlantUML code from response text."""
                # Try to find code block first
                m = re.search(r"```plantuml\s*(@startuml[\s\S]*?@enduml)\s*```", response_text, re.IGNORECASE)
                if m:
                    return m.group(1).strip()
                
                # Try to find @startuml...@enduml directly
                m = re.search(r"@startuml\s*([\s\S]*?)\s*@enduml", response_text)
                if m:
                    return f"@startuml\n{m.group(1).strip()}\n@enduml"
                
                # If no PlantUML found, return the whole response
                logging.warning("No PlantUML code block found in response.")
                return response_text        
        def generate(self, system_description, context):
            
            system_prompt = """
                            You're a practical and experienced software architect. Your job is to translate a system description into a clean, readable, and syntactically correct PlantUML class diagram. Focus on clarity, correctness, and matching the described system as directly as possible.
                            ## Step by Step to create a class diagram

                            Before creating the class diagram, follow this systematic analysis:

                            ### Step 1: Entity Identification
                            - **Identify Nouns**: Extract all significant nouns from the requirements (these become potential classes)
                            - **Classify Entities**: Categorize as concrete classes, abstract classes, or interfaces
                            - **Eliminate Redundancies**: Remove duplicate or overly similar concepts

                            ### Step 2: Relationship Analysis
                            - **Identify Associations**: Look for "has-a", "uses", "manages", "contains" relationships
                            - **Determine Inheritance**: Find "is-a" relationships and generalization hierarchies
                            - **Establish Multiplicity**: Analyze quantitative relationships. Always specify multiplicities (e.g., `"1" *-- "1..*"`).
                            - **Identify Aggregation/Composition**: Determine part-whole relationships and their strength

                            ### Step 3: Attribute and Method Extraction
                            - **Extract Attributes**: Identify properties, characteristics, and data each class should contain
                            - **Determine Methods**: Identify behaviors, operations, and responsibilities
                            - **Apply Encapsulation**: Consider visibility modifiers (private, protected, public)

                            ### Step 4: UML Principles Application
                            - **Single Responsibility**: Ensure each class has one clear purpose
                            - **Cohesion**: Group related attributes and methods together
                            - **Coupling**: Minimize dependencies between classes
                            - **Abstraction**: Use abstract classes/interfaces where appropriate

                            ### Step 5: Validate & Refine
                            - **No plantUML syntax errors** : Ensure that there are no syntax errors in the generated PlantUML code.
                            - **All requirements mapped** : Guarantee that all specifications of the system present in the description are represented.

                            ## PlantUML Syntax Guidelines

                            ### Class Declaration
                            class ClassName {
                                - privateAttribute: Type
                                # protectedAttribute: Type
                                + publicAttribute: Type
                                --
                                - privateMethod(): ReturnType
                                + publicMethod(param: Type): ReturnType
                                {abstract} abstractMethod(): ReturnType
                                {static} staticMethod(): Type
                            }

                            abstract class AbstractClassName
                            interface InterfaceName

                            ### Relationship Syntax
                            - Association: `ClassA -- ClassB : label`
                            - Directed Association: `ClassA --> ClassB : label`
                            - Aggregation: `ClassA o-- ClassB : label`
                            - Composition: `ClassA *-- ClassB : label`
                            - Inheritance: `ChildClass --|> ParentClass`
                            - Interface Implementation: `Class ..|> Interface`
                            - Dependency: `ClassA ..> ClassB : uses`

                            ### Multiplicity Notation
                            - "1" - exactly one
                            - "0..1" - zero or one
                            - "1..*" - one or more
                            - "0..*" - zero or more
                            - "n" - exactly n
                            - "m..n" - between m and n

                            ### Advanced PlantUML Features
                            - Packages: `package PackageName { ... }`
                            - Stereotypes: `<<stereotype>>`

                            ## Output Requirements

                            1. **Structure**: Begin with `@startuml` and end with `@enduml`
                            2. **Naming**: Use PascalCase for classes, camelCase for attributes/methods
                            3. **Organization**: Group related classes logically
                            4. **Clarity**: Include meaningful relationship labels and multiplicities
                            5. **Completeness**: Represent all significant entities and relationships from requirements
                            
                            ##Important Note: 
                            Follow strictly the System Description. Keep the diagram simple and direct.

                            
                            """

            user_prompt = "Generate a class diagram based on the following system description and Context: \n ##System Description \n" + system_description + "\n"  "## Context \n" + context 
            try:
                # Call Anthropic API
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                response = self.clientContext.invoke(messages)
                response_text = response.content

                # Extract PlantUML code
                plantuml_code = self._extract_plantuml_code(response_text)
                logging.info(f"Generated PlantUML code:\n{plantuml_code}")
                #message_history.extend( [{"role": "user", "content": user_prompt},{"role": "assistant", "content": plantuml_code}])
                
                return plantuml_code          
            except Exception as e:
                logging.error(f"Error calling Anthropic API: {e}")
                return f"Error generating class diagram: {e}"

   
class SelfRefinement:
    """Self-Refinement RAG that evaluates and improves its own outputs."""
    logging.basicConfig(
        filename="SelfRefinement.log",   # where to write
        filemode="w",                        # overwrite each run ("a" to append)
        level=logging.INFO,                  # capture INFO and above
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    def __init__(self, anthropic_api_key: str, initialDiagram:str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.message_history = []
        self.initialDiagram = initialDiagram
    def stringify_message_history(self, message_history):
        if not message_history:
            return ""

        # Try to get the last two messages: user and assistant
        last_messages = message_history[-2:] if len(message_history) >= 2 else message_history
        return "\n\n".join([
            f"{msg['role'].capitalize()}:\n{msg['content']}"
            for msg in last_messages
        ])

    async def generate_with_reflection(self, query: str, max_iterations: int = 3) -> str:
        """Generate response with self-reflection and improvement."""
        
        current_response = self.initialDiagram
        
        for iteration in range(max_iterations):
            logging.info(f"------------ Iteration {iteration}------------------")
            
            # Self-evaluate
            feedback = await self._self_evaluate(query, current_response)
            
            #Improve the current response
            current_response = await self._improve_response(query, current_response, feedback)
            
            if feedback == {}: break
           
            # Check if satisfactory
            if iteration > 0 and abs(feedback["overall_score"] - prev_score) < 0.01:
                logging.info(f"Satisfactory quality achieved at iteration {iteration + 1}")
                break
            
            logging.info(f"Iteration {iteration}: Overall score {feedback.get('overall_score', 0):.2f}")
            prev_score = feedback["overall_score"]

        return current_response
    
    async def _improve_response(self, system_description: str, current_response: str, feedback: dict) -> str:
        """Improve the class diagram based on feedback."""
        
        improve_system_prompt = f"""
        You're an experienced software design mentor reviewing a junior architect's class diagram. You've received feedback that highlights specific issues with the diagram, and it's your job to revise it accordingly.

        Use the feedback constructively. Your goal is to make the diagram more complete, clearer, and more aligned with UML best practices—without deviating from the system description.

        ## Your Task
        Analyze the provided feedback and systematically improve the class diagram to address all identified issues while maintaining the original system requirements.

        ## Improvement Guidelines

        ### 1. UML Completeness Improvements
        - Add missing classes, attributes, or methods identified in feedback
        - Ensure all entities from system description are represented
        - Include proper visibility modifiers (-, +, #)
        - Add missing relationships and associations

        ### 2. UML Quality Enhancements
        - Fix naming convention issues (PascalCase for classes, camelCase for attributes/methods)
        - Correct relationship types (association, aggregation, composition, inheritance)
        - Improve multiplicity specifications
        - Enhance method signatures with proper parameters and return types

        ### 3. Requirement Coverage Improvements
        - Map every requirement explicitly mentioned in system description
        - Ensure behavioral aspects are captured through appropriate methods
        - Add missing domain-specific attributes and operations
        - Include all stated constraints and business rules

        ### 4. Clarity & Pragmatic Improvements
        - Improve relationship labels for better understanding
        - Organize classes logically (group related classes)
        - Add stereotypes where appropriate (<<interface>>, <<abstract>>, etc.)
        - Ensure proper abstraction levels

        ## Output Instructions
        1. Provide the complete improved PlantUML diagram
        2. Start with @startuml and end with @enduml
        3. Include all improvements addressing the feedback
        4. Maintain syntactic correctness
        5. Ensure the diagram is more comprehensive than the original

        ## Critical Requirements
        - Address ALL issues mentioned in the feedback
        - Implement ALL recommendations provided
        - Ensure the improved diagram scores higher in all evaluation dimensions
        - Maintain or enhance requirement coverage
        - Keep PlantUML syntax error-free

        ##Important Note: 
        - Follow strictly the System Description. Keep the diagram simple and direct.

       """
        
        improve_user_prompt = f""" 
        Previous feedback and responses: {self.stringify_message_history(self.message_history)} 
        
        Improve the following PlantUML class diagram based on the following feedback:      
            ## Current PlantUML Class Diagram
            {current_response}

            ## Detailed Feedback Analysis
            {json.dumps(feedback.get('feedback', ''), indent=2)}

            ## Specific Issues to Address
            {chr(10).join(f"- {issue}" for issue in feedback.get('issues', []))}

            ## Recommendations to Implement
            {chr(10).join(f"- {rec}" for rec in feedback.get('recommendations', []))}
        """
        try:
           
            # Call Anthropic API for improvement
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=2000,
                temperature=0.5,
                system= improve_system_prompt,
                messages=[
                    {"role": "user", "content": improve_user_prompt},
                   
                ]
            )
            
            response_text = response.content[0].text
            
            improved_plantuml = self._extract_plantuml_code(response_text)
            
            logging.info(f"Class Diagram improved:\n{improved_plantuml}")
            self.message_history.extend([
                {"role": "user", "content": improve_user_prompt},
                {"role": "assistant", "content": improved_plantuml}]
            )
            return improved_plantuml
            
        except Exception as e:
            logging.error(f"Error improving response: {e}")
            return current_response  # Return original if improvement fails
    
    def _extract_plantuml_code(self, response_text: str) -> str:
        """Extract PlantUML code from response text."""
        # Try to find code block first
        m = re.search(r"```plantuml\s*(@startuml[\s\S]*?@enduml)\s*```", response_text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        
        # Try to find @startuml...@enduml directly
        m = re.search(r"@startuml\s*([\s\S]*?)\s*@enduml", response_text)
        if m:
            return f"@startuml\n{m.group(1).strip()}\n@enduml"
        
        # If no PlantUML found, return the whole response
        logging.warning("No PlantUML code block found in response.")
        return response_text
    
    async def _self_evaluate(self, system_description: str, response: str) -> dict:
        """Enhanced self-evaluation of the generated response."""
        
        evaluation_system_prompt = f"""
            You are acting as a formal academic reviewer with expertise in software architecture and modeling. Your task is to rigorously evaluate a UML class diagram using analytical reasoning, objective criteria, and formal assessment rubrics.

            Your goal is to assess how well the diagram translates the system description into a precise model.

            ## Evaluation Framework

            Assess the class diagram across these four critical dimensions:

            ### 1. UML Completeness (25% weight)
            **Criteria:**
            - All relevant classes identified and included
            - Complete set of attributes for each class with appropriate types
            - Comprehensive method signatures with parameters and return types
            - Proper visibility modifiers (-, +, #) applied consistently
            - All necessary relationships present (associations, inheritance, dependencies)

            **Scoring:**
            - 0.9-1.0: All elements present, comprehensive coverage
            - 0.7-0.8: Minor omissions, mostly complete
            - 0.5-0.6: Several missing elements
            - 0.3-0.4: Major gaps in completeness
            - 0.0-0.2: Severely incomplete

            ### 2. UML Quality (25% weight)
            **Criteria:**
            - Correct UML syntax and notation
            - Proper relationship types (composition vs aggregation vs association)
            - Accurate multiplicity specifications
            - Consistent naming conventions (PascalCase classes, camelCase attributes/methods)
            - Appropriate use of abstract classes and interfaces
            - Proper encapsulation and information hiding

            **Scoring:**
            - 0.9-1.0: Excellent quality, best practices followed
            - 0.7-0.8: Good quality with minor issues
            - 0.5-0.6: Adequate quality, some problems
            - 0.3-0.4: Poor quality, multiple issues
            - 0.0-0.2: Very poor quality, major problems

            ### 3. Requirement Coverage (30% weight)
            **Criteria:**
            - Every entity mentioned in requirements is modeled
            - All relationships described in requirements are captured
            - Business rules and constraints are represented
            - Behavioral aspects are included through appropriate methods
            - Domain-specific terminology is preserved
            - System boundaries and scope are respected

            **Scoring:**
            - 0.9-1.0: Complete requirement coverage
            - 0.7-0.8: Most requirements covered, minor gaps
            - 0.5-0.6: Adequate coverage, some missing aspects
            - 0.3-0.4: Poor coverage, major requirements missing
            - 0.0-0.2: Very poor coverage, most requirements ignored

            ### 4. Clarity & Pragmatics (20% weight)
            **Criteria:**
            - Diagram is understandable to technical stakeholders
            - Logical organization and grouping of classes
            - Meaningful relationship labels and role names
            - Appropriate level of abstraction
            - Clear separation of concerns
            - Professional presentation and layout

            **Scoring:**
            - 0.9-1.0: Exceptionally clear and well-organized
            - 0.7-0.8: Clear and understandable
            - 0.5-0.6: Reasonably clear, some confusion possible
            - 0.3-0.4: Unclear in several areas
            - 0.0-0.2: Very unclear, difficult to understand

            ## Evaluation Instructions

            1. **Systematic Analysis**: Examine each dimension thoroughly
            2. **Evidence-Based Scoring**: Provide specific examples for scores
            3. **Detailed Issues**: List concrete problems with specific locations
            4. **Actionable Recommendations**: Provide implementable improvements
            5. **Comprehensive Assessment**: Consider the diagram as a whole system model

            ## Required Output Format

            Return ONLY a valid JSON object with this exact structure:

            ## Important Note: In case you don't have an issues or  recommendations to give just return an empty array for those fields.
            ```json
            {{
                "uml_completeness_score": <float 0.0-1.0>,
                "uml_quality_score": <float 0.0-1.0>,
                "requirement_coverage_score": <float 0.0-1.0>,
                "pragmatic_clarity_score": <float 0.0-1.0>,
                "overall_score": <float 0.0-1.0>,
                "detailed_analysis": {{
                    "completeness_details": "<specific analysis of what's missing or present>",
                    "quality_details": "<analysis of UML syntax, conventions, and design quality>",
                    "coverage_details": "<analysis of how well requirements are captured>",
                    "clarity_details": "<analysis of diagram understandability and organization>"
                }},
                "issues": [
                    "<specific issue 1 with location/context>",
                    "<specific issue 2 with location/context>",
                    "<specific issue 3 with location/context>"
                ],
                "recommendations": [
                    "<actionable recommendation 1>",
                    "<actionable recommendation 2>",
                    "<actionable recommendation 3>"
                ],
                "strengths": [
                    "<identified strength 1>",
                    "<identified strength 2>"
                ]
            }}
            ```
            If there is no issues or recommendations, you can return an empty array for those fields.
            Calculate the overall_score as: (uml_completeness_score * 0.25) + (uml_quality_score * 0.25) + (requirement_coverage_score * 0.30) + (pragmatic_clarity_score * 0.20)"""
        
        evaluation_user_prompt = f"""   
            Previous feedback and responses: {self.stringify_message_history(self.message_history)}

            Evaluate the following PlantUML class diagram based on the following system description:

            **System Description**
            {system_description}

            **PlantUML class diagram**
            {response}
            Base your response exclusively on the system description and in the UML principles.
        """
 
        try:
        
            eval_response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=2000,
                temperature=0.2,
                system= evaluation_system_prompt,
                messages=[{"role": "user", "content": evaluation_user_prompt}]
            )

            if not eval_response.content:
                return { }
            
            eval_text = eval_response.content[0].text

            # Extract JSON from response
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", eval_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code block
                json_match = re.search(r"\{[\s\S]*\}", eval_text)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = eval_text
            
            try:
                evaluation_result = json.loads(json_str)
                self.message_history.extend([
                    {"role": "user", "content": evaluation_user_prompt},
                    {"role": "assistant", "content": json.dumps(evaluation_result, indent=2)}]
                )
                logging.info(f"Self-evaluation result: {json.dumps(evaluation_result, indent=2)}")
                return evaluation_result
            
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse evaluation JSON: {e}")
                logging.error(f"Raw evaluation text: {eval_text}")
                
                # Fallback evaluation
                return {
                    'uml_completeness_score': 0.6,
                    'uml_quality_score': 0.6,
                    'requirement_coverage_score': 0.6,
                    'pragmatic_clarity_score': 0.6,
                    'overall_score': 0.6,
                    'issues': ['Evaluation parsing failed'],
                    'recommendations': ['Review and fix evaluation system'],
                    'feedback': eval_text
                }
        
        except Exception as e:
            logging.error(f"Self-evaluation failed: {e}")
            return {
                'uml_completeness_score': 0.5,
                'uml_quality_score': 0.5,
                'requirement_coverage_score': 0.5,
                'pragmatic_clarity_score': 0.5,
                'overall_score': 0.5,
                'issues': ['Evaluation system error'],
                'recommendations': ['Fix evaluation system'],
                'feedback': f'Evaluation failed: {e}'
            }

class RenderDiagram:
    def __init__(self,fileName):
        self.filename = fileName
    
    
    def render_plantuml_locally(plantuml_code: str) -> str:
        # Unique filename to avoid collision
        file_id = uuid.uuid4().hex
        base_dir = "./uml_outputs"
        os.makedirs(base_dir, exist_ok=True)

        puml_path = os.path.join(base_dir, f"{file_id}.puml")
        png_path = os.path.join(base_dir, f"{file_id}.png")

        # Write PlantUML code to file
        with open(puml_path, "w") as f:
            f.write(plantuml_code)

        # Run PlantUML via Java
        result = subprocess.run(
            ["java", "-jar", "plantuml-1.2025.3.jar", puml_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            print("Error rendering UML: %s", result.stderr.decode('utf-8'))
            return f"Error rendering UML: {result.stderr.decode('utf-8')}"
        print("UML diagram rendered successfully")
        return f"UML diagram saved at: {png_path}"

        
async def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
    
    rag = RAG()


    system_description = """
    A project manager uses the project management system to manage a project. The project manager leads a team to execute the project within the project's start and end dates. Once a project is created in the project management system, a manager may initiate and later terminate the project due to its completion or for some other reason. As input, a project uses requirements. As output, a project produces a system (or part of a system). The requirements and system are work products: things that are created, used, updated, and elaborated on throughout a project. Every work product has a description, is of some percent complete throughout the effort, and may be validated. However, validation is dependent on the type of work product. For example, the requirements are validated with users in workshops, and the system is validated by being tested against the requirements. Furthermore, requirements may be published using various types of media, including on an intranet or in paper form; and systems may be deployed onto specific platforms    """
    
    context_generated1 = rag.generate_results()
    context_generated2 = rag.retrieve_context(system_description)

    print(f"First context generated {context_generated1}\n")
    print(f"Second context generated {context_generated2}\n")

    PlantUMLCode = rag.generate( system_description, context_generated1 )
    PlantUMLCode2 = rag.generate( system_description,  context_generated2)
    
    
    print(f"This was the generated class diagram using the first context {PlantUMLCode}")
    print(f"This was the generated class diagram using the second context {PlantUMLCode2}")

    self_refinement = SelfRefinement(api_key, PlantUMLCode)
    refinedClassDiagram = self_refinement.generate_with_reflection(system_description)




if __name__ == "__main__":
    asyncio.run(main())