"""
Flask API wrapper for the FinalImplementation.py diagram generation system
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, InternalServerError

# Import your existing classes
from FinalImplementation import RAG, SelfRefinement, RenderDiagram

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configure logging
logging.basicConfig(
    filename = "api.log",
    filemode = "w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    #handlers=[logging.FileHandler("api.log"), logging.StreamHandler()]
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Global instances (initialize once)
rag_instance = None
base_output_dir = "./uml_outputs"

def initialize_rag():
    """Initialize RAG instance on first use"""
    global rag_instance
    if rag_instance is None:
        try:
            logging.info("Initializing RAG system...")
            rag_instance = RAG()
            logging.info("RAG system initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize RAG: {e}")
            raise e
    return rag_instance

def render_plantuml_locally(plantuml_code: str) -> tuple:
    """
    Render PlantUML code to PNG image
    Returns: (success: bool, file_path_or_error: str)
    """
    try:
        import subprocess
        
        # Create unique filename
        file_id = uuid.uuid4().hex
        os.makedirs(base_output_dir, exist_ok=True)
        
        puml_path = os.path.join(base_output_dir, f"{file_id}.puml")
        png_path = os.path.join(base_output_dir, f"{file_id}.png")
        
        # Write PlantUML code to file
        with open(puml_path, "w", encoding='utf-8') as f:
            f.write(plantuml_code)
        
        # Run PlantUML via Java (make sure plantuml jar is in the same directory)
        result = subprocess.run(
            ["java", "-jar", "plantuml-1.2025.3.jar", puml_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="."  # Current directory where jar file should be
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8')
            logging.error(f"PlantUML rendering error: {error_msg}")
            return False, f"PlantUML rendering failed: {error_msg}"
        
        # Check if PNG file was created
        if not os.path.exists(png_path):
            return False, "PNG file was not generated"
        
        logging.info(f"PlantUML diagram rendered successfully: {png_path}")
        return True, png_path
        
    except Exception as e:
        logging.error(f"Error in render_plantuml_locally: {e}")
        return False, str(e)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "Diagram generation API is running"
    })

@app.route('/api/generate-diagram', methods=['POST'])
def generate_diagram():
    """
    Main endpoint for diagram generation
    Expected input: {"diagram_type": "class", "system_description": "..."}
    """
    try:
        # Parse request
        if not request.is_json:
            raise BadRequest("Request must be JSON")
        
        data = request.get_json()
        
        # Validate input
        if not data.get('system_description'):
            raise BadRequest("system_description is required")
        
        diagram_type = data.get('diagram_type', 'class')
        system_description = data.get('system_description').strip()
        
        logging.info(f"Generating {diagram_type} diagram for system: {system_description[:100]}...")
        
        # Initialize RAG
        rag = initialize_rag()
        if not rag.loaded:
            rag.load()
        # Generate contexts
        logging.info("Generating RAG contexts...")
        context_generated1 = rag.generate_results()
        context_generated2 = rag.retrieve_context(system_description)
        
        # Generate initial PlantUML code
        logging.info("Generating initial PlantUML code...")
        initial_plantuml = rag.generate(system_description, context_generated1)
        
        # Apply self-refinement
        logging.info("Applying self-refinement...")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise InternalServerError("ANTHROPIC_API_KEY not configured")
        
        self_refinement = SelfRefinement(api_key, initial_plantuml)
        
        # Run async self-refinement
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            refined_plantuml = loop.run_until_complete(
                self_refinement.generate_with_reflection(system_description, max_iterations=2)
            )
        finally:
            loop.close()
        
        # Render to PNG
        logging.info("Rendering PlantUML to PNG...")
        success, result = render_plantuml_locally(refined_plantuml)
        
        if not success:
            raise InternalServerError(f"Failed to render diagram: {result}")
        
        png_path = result
        
        # Create response
        response_data = {
            "success": True,
            "diagram_type": diagram_type,
            "diagram_url": f"/api/diagram/{os.path.basename(png_path)}",
            "puml_content": refined_plantuml,
            "message": "Diagram generated successfully",
            "timestamp": datetime.now().isoformat()
        }
        
        logging.info("Diagram generation completed successfully")
        return jsonify(response_data)
        
    except BadRequest as e:
        logging.warning(f"Bad request: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 400
        
    except Exception as e:
        logging.error(f"Internal error in generate_diagram: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/diagram/<filename>', methods=['GET'])
def serve_diagram(filename):
    """Serve generated diagram images"""
    try:
        file_path = os.path.join(base_output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "Diagram not found"}), 404
        
        if not filename.endswith('.png'):
            return jsonify({"error": "Invalid file type"}), 400
        
        return send_file(file_path, mimetype='image/png')
        
    except Exception as e:
        logging.error(f"Error serving diagram {filename}: {e}")
        return jsonify({"error": "Failed to serve diagram"}), 500

@app.route('/api/diagrams', methods=['GET'])
def list_diagrams():
    """List all generated diagrams"""
    try:
        if not os.path.exists(base_output_dir):
            return jsonify({"diagrams": []})
        
        diagrams = []
        for filename in os.listdir(base_output_dir):
            if filename.endswith('.png'):
                file_path = os.path.join(base_output_dir, filename)
                stat = os.stat(file_path)
                diagrams.append({
                    "filename": filename,
                    "url": f"/api/diagram/{filename}",
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "size": stat.st_size
                })
        
        diagrams.sort(key=lambda x: x['created'], reverse=True)
        return jsonify({"diagrams": diagrams})
        
    except Exception as e:
        logging.error(f"Error listing diagrams: {e}")
        return jsonify({"error": "Failed to list diagrams"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Verify required files
    if not os.path.exists("plantuml-1.2025.3.jar"):
        print("plantuml-1.2025.3.jar not found in current directory")
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY environment variable not set")
        exit(1)
    
    print("Flask server ready. RAG will initialize on first request.")
    
    # Run Flask app
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,  # Set to False in production
        threaded=True
    )
