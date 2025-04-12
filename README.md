# Knowledge Graph Explorer: Interactive Graph Traversal & QA System

![Knowledge Graph](https://img.shields.io/badge/Knowledge-Graph-blue)
![Neo4j](https://img.shields.io/badge/Database-Neo4j-brightgreen)
![OpenAI](https://img.shields.io/badge/LLM-OpenAI-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

A sophisticated knowledge graph-based question answering system that learns, recalls, and reasons across interconnected concepts.

## Overview

Knowledge Graph Explorer is an intelligent QA system that combines a Neo4j graph database with LLM capabilities to traverse knowledge connections, answer questions, find analogies, generate ideas, and explore concept spaces. The system features an interactive Streamlit web interface with rich visualizations of knowledge paths and concept relationships.

Key features:
- 🧠 **Factual Q&A**: Find answers based on structured knowledge
- 🔄 **Analogical Reasoning**: Discover meaningful connections between concepts
- 💡 **Idea Generation**: Create novel combinations of existing knowledge
- 🔍 **Knowledge Exploration**: Traverse concept spaces and visualize relationships
- 📚 **Continuous Learning**: Automatically extend knowledge from conversations and explicit learning commands

## System Architecture

The Cognitive Recall system focuses on two main aspects:

### 1. Knowledge Graph QA Module
- `knowledge_graph_qa.py`: Core QA functionality
- `knowledge_graph_qa_extension.py`: Enhanced reasoning capabilities
- `knowledge_graph_traversal.py`: Graph path traversal engine
- `knowledge_graph_path_scoring.py`: Relevance scoring for knowledge paths

### 2. Visualization & Interface
- `app.py`: Streamlit web application
- `streamlit_visualization.py`: Interactive knowledge graph visualizations

### User Interfaces
- `app.py`: Streamlit web application
- `streamlit_visualization.py`: Interactive visualizations

## Setup & Installation

### Prerequisites
- Python 3.8+
- Neo4j Database (4.4+)
- OpenAI API key

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cognitive-recall.git
   cd cognitive-recall
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   Key dependencies include:
   - streamlit
   - neo4j
   - langchain
   - langchain-openai
   - openai
   - python-dotenv
   - scikit-learn

3. Set up environment variables (create a `.env` file):
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   OPENAI_API_KEY=your_openai_key
   OPENAI_MODEL=gpt-4o-mini
   ```

## Running the Application

### Streamlit Web App

Start the Streamlit web interface:

```bash
streamlit run app.py
```

The web interface will be available at `http://localhost:8501`.



```

## Knowledge Graph Structure

### Concept Node Properties
- `name`: Primary identifier
- `what`: Definition/description
- `how`: How it works/is used
- `why`: Purpose/significance
- `when`: Temporal context
- `domain`: Knowledge domain
- `type`: Concept type
- `aliases`: Alternative names
- `embedding`: Vector representation

### Relationship Types
- Dynamic relationship types based on semantic meaning
- Includes justification for each relationship

## Query Types

The system supports five types of queries:

1. **Factual**: Retrieve specific information
   ```
   What is ESP32?
   ```

2. **Analogy**: Find connections between concepts
   ```
   What's the relationship between ESP32 and WiFi?
   ```

3. **Idea Generation**: Create novel combinations
   ```
   Generate ideas combining machine learning and Bengali cuisine
   ```

4. **Comparison**: Compare multiple concepts
   ```
   Compare ESP32 and STM32
   ```

5. **Exploration**: Explore concept spaces
   ```
   Explore the concept of ESP32
   ```

## Visualization

The system provides interactive visualizations:
- Knowledge path graphs
- Concept clusters
- Analysis results including ideas, analogies, and explorations

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

[MIT License](LICENSE)

## Acknowledgements

- Neo4j for graph database technology
- OpenAI for LLM capabilities
- Streamlit for the web interface
