import React, { useState, useRef } from 'react';
import { FileText, Database, Workflow, GitBranch, Users, Clock, AlertCircle, CheckCircle, Loader2, Clipboard,ClipboardCheck } from 'lucide-react';

const DiagramGeneratorInterface = () => {
  const [selectedDiagram, setSelectedDiagram] = useState('');
  const [systemDescription, setSystemDescription] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const codeRef = useRef<HTMLElement | null>(null);

  type DiagramResult = {
  success: boolean;
  diagram_url?: string;
  puml_content?: string;
  message?: string;
  diagram_type?: string;
  };

  const [result, setResult] = useState<DiagramResult | null>(null);

  const [error, setError] = useState('');

  // Diagram types with icons and descriptions
  const diagramTypes = [
    {
      id: 'class',
      name: 'Class Diagram', 
      icon: <Database className="w-6 h-6" />,
      description: 'Display class structure and relationships using AI-powered RAG and self-refinement'
    },
    {
      id: 'sequence',
      name: 'Sequence Diagram',
      icon: <Clock className="w-6 h-6" />,
      description: 'Show interactions between objects over time (Coming Soon)'
    },
    {
      id: 'activity',
      name: 'Activity Diagram',
      icon: <Workflow className="w-6 h-6" />,
      description: 'Visualize workflow and business processes (Coming Soon)'
    },
    {
      id: 'usecase',
      name: 'Use Case Diagram',
      icon: <Users className="w-6 h-6" />,
      description: 'Show system functionality from user perspective (Coming Soon)'
    },
    {
      id: 'component',
      name: 'Component Diagram',
      icon: <GitBranch className="w-6 h-6" />,
      description: 'Illustrate component organization and dependencies (Coming Soon)'
    }
  ];

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement> | React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    
    // Validation
    if (!selectedDiagram || !systemDescription.trim()) {
      setError('Please select a diagram type and provide a system description');
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      // Call your Flask API endpoint
      const response = await fetch('http://127.0.0.1:5000/api/generate-diagram', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          diagram_type: selectedDiagram,
          system_description: systemDescription
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Expected response format from your Python API:
      // {
      //   "success": true,
      //   "diagram_url": "/api/diagram/filename.png", 
      //   "puml_content": "PlantUML source code",
      //   "message": "Diagram generated successfully",
      //   "diagram_type": "class"
      // }
      
      setResult(data);
   } catch (err: unknown) {
      let errorMessage = 'Failed to generate diagram';
      if (err instanceof Error) {
        if (err.message.includes('fetch')) {
          errorMessage = 'Cannot connect to the diagram generation service. Please ensure the Flask API is running on http://127.0.0.1:5000';
        } else {
          errorMessage = err.message;
        }
      }

      setError(errorMessage);
      console.error('Error generating diagram:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Clear form and results
  const handleClear = () => {
    setSelectedDiagram('');
    setSystemDescription('');
    setResult(null);
    setError('');
  };
  const copyPumlToClipboard = async () => {
  const text = result?.puml_content ?? '';
  try {
    // visually select everything
    if (codeRef.current) {
      const range = document.createRange();
      range.selectNodeContents(codeRef.current);
      const sel = window.getSelection();
      if (sel) {
        sel.removeAllRanges();
        sel.addRange(range);
      }
    }
    // copy to clipboard
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  } catch (e) {
    setError('Failed to copy PlantUML to clipboard');
  }
};


  return (
    <div className="min-h-screen bg-gray-100 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Diagram Generator
          </h1>
          <p className="text-gray-600">
            Generate PlantUML diagrams from system descriptions using AI
          </p>
        </div>

        {/* Main Interface */}
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          {/* Form Section */}
          <div className="p-6 border-b border-gray-200">
            <div className="space-y-6">
              {/* Diagram Type Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Select Diagram Type
                </label>

                <div className="bg-white rounded-lg shadow border border-gray-200 p-6">
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
                    {diagramTypes.map((type) => {
                      const selected = selectedDiagram === type.id;
                      return (
                        <button
                          key={type.id}
                          onClick={() => setSelectedDiagram(type.id)}
                          className={`flex flex-col items-start text-left border rounded-xl p-4 transition-all duration-200 shadow-sm hover:shadow-md focus:outline-none ${
                            selected
                              ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                              : 'border-gray-200 bg-white'
                          }`}
                        >
                          <div className="flex items-center space-x-3 mb-2">
                            <div className={`text-xl ${selected ? 'text-blue-600' : 'text-gray-400'}`}>
                              {type.icon}
                            </div>
                            <h3 className={`text-base font-semibold ${selected ? 'text-blue-900' : 'text-gray-900'}`}>
                              {type.name}
                            </h3>
                          </div>
                          <p className={`text-sm ${selected ? 'text-blue-700' : 'text-gray-500'}`}>
                            {type.description}
                          </p>
                        </button>
                      );
                    })}
                  </div>
                </div>
            </div>
              {/* System Description Input */}
              <div>
                <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-2">
                  System Description
                </label>
                <textarea
                  id="description"
                  value={systemDescription}
                  onChange={(e) => setSystemDescription(e.target.value)}
                  placeholder="Describe your system, components, interactions, or workflow here. Example: A project manager uses the project management system to manage a project. The project manager leads a team to execute the project within the project's start and end dates..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 resize-vertical"
                  rows={6}
                  disabled={isLoading}
                />
                <p className="mt-1 text-sm text-gray-500">
                  Provide a detailed description of your system. The AI will use RAG and self-refinement to generate an optimized PlantUML class diagram.
                </p>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-3">
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={isLoading || !selectedDiagram || !systemDescription.trim()}
                  className="flex-1 flex items-center justify-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <FileText className="w-4 h-4 mr-2" />
                      Generate Diagram
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={handleClear}
                  disabled={isLoading}
                  className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Clear
                </button>
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="p-6">
            {/* Error Display */}
            {error && (
              <div className="mb-6 p-4 border border-red-200 rounded-md bg-red-50">
                <div className="flex items-center">
                  <AlertCircle className="w-5 h-5 text-red-600 mr-2" />
                  <span className="text-sm font-medium text-red-800">Error</span>
                </div>
                <p className="mt-1 text-sm text-red-700">{error}</p>
              </div>
            )}

            {/* Success Result */}
            {result && result.success && (
              <div className="space-y-4">
                <div className="flex items-center text-green-700 mb-4">
                  <CheckCircle className="w-5 h-5 mr-2" />
                  <span className="font-medium">Diagram Generated Successfully</span>
                </div>
                
                {/* Generated Diagram Image */}
                {result.diagram_url && (
                  <div className="border border-gray-200 rounded-lg overflow-hidden">
                    <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
                      <h3 className="text-sm font-medium text-gray-900">Generated Diagram</h3>
                    </div>
                    <div className="p-4 bg-white">
                      <img
                        src={`http://127.0.0.1:5000${result.diagram_url}`}
                        alt="Generated diagram"
                        className="max-w-full h-auto mx-auto border border-gray-200 rounded"
                        onError={(e) => {
                           (e.target as HTMLImageElement).style.display = 'none';
                          setError('Failed to load generated diagram image');
                        }}
                      />
                    </div>
                  </div>
                )}

                {/* PlantUML Source (collapsible) */}
                {result.puml_content && (
                    <details className="border border-gray-200 rounded-lg">
                    <summary className="bg-gray-50 px-4 py-2 cursor-pointer text-sm font-medium text-gray-900 hover:bg-gray-100 transition-colors flex items-center justify-between">
                      <span>View PlantUML Source Code</span>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();      // don't toggle the <details>
                          e.stopPropagation();     // keep it open/closed as-is
                          copyPumlToClipboard();
                        }}
                        className="inline-flex items-center gap-2 rounded-md border border-gray-300 bg-white px-3 py-1.5 text-xs font-medium text-gray-700 hover:bg-gray-50"
                        aria-label="Copy PlantUML to clipboard"
                        title="Copy PlantUML to clipboard"
                      >
                        {copied ? <ClipboardCheck className="w-4 h-4" /> : <Clipboard className="w-4 h-4" />}
                        {copied ? 'Copied' : 'Copy'}
                      </button>
                    </summary>

                    <div className="p-4 bg-white">
                      <pre className="text-sm text-gray-800 bg-gray-50 p-3 rounded border overflow-x-auto">
                        {/* ref is used to select all text visually */}
                        <code ref={codeRef}>{result.puml_content}</code>
                      </pre>
                    </div>
                  </details>
                )}
              </div>
            )}

            {/* Loading State */}
            {isLoading && (
              <div className="text-center py-8">
                <Loader2 className="w-8 h-8 animate-spin mx-auto text-blue-600 mb-2" />
                <p className="text-gray-600">Processing your request...</p>
                <p className="text-sm text-gray-500 mt-1">
                  Generating diagram using AI, RAG, and self-refinement (~30-60 seconds)
                </p>
                <div className="mt-4 text-xs text-gray-400 space-y-1">
                  <p>• Analyzing system description with RAG...</p>
                  <p>• Generating initial PlantUML code...</p>
                  <p>• Applying self-refinement iterations...</p>
                  <p>• Rendering final diagram...</p>
                </div>
              </div>
            )}

            {/* Empty State */}
            {!result && !error && !isLoading && (
              <div className="text-center py-8 text-gray-500">
                <FileText className="w-12 h-12 mx-auto mb-3 text-gray-400" />
                <p>Select a diagram type and provide a system description to get started</p>
              </div>
            )}
          </div>
        </div>

        {/* Integration Instructions */}
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-lg font-medium text-blue-900 mb-3">
            Setup Instructions
          </h3>
          <div className="text-sm text-blue-800 space-y-2">
            <p><strong>1. Install Dependencies:</strong> <code>pip install flask flask-cors python-dotenv</code></p>
            <p><strong>2. Environment Setup:</strong> Create <code>.env</code> file with <code>ANTHROPIC_API_KEY=your_key_here</code></p>
            <p><strong>3. PlantUML JAR:</strong> Ensure <code>plantuml-1.2025.3.jar</code> is in the same directory</p>
            <p><strong>4. Start API:</strong> <code>python flask_api.py</code> (runs on http://127.0.0.1:5000)</p>
            <p><strong>5. Current Status:</strong> Only Class Diagrams are fully implemented with RAG + Self-Refinement</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiagramGeneratorInterface;