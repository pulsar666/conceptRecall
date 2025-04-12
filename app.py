import streamlit as st
import os
from dotenv import load_dotenv
import json
from enum import Enum
import time

# Import the knowledge graph QA module
from knowledge_graph_qa import KnowledgeGraphQA, QueryType, KnowledgeGraphQuery
from knowledge_graph_qa import QueryType, KnowledgeGraphQuery
from knowledge_graph_qa_extension import EnhancedKnowledgeGraphQA
from streamlit_visualization import render_enhanced_query_results, render_knowledge_paths

# Load environment variables
load_dotenv()

# Get Neo4j connection details from environment or defaults
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j123")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Page configuration
st.set_page_config(
    page_title="Knowledge Graph QA",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_system" not in st.session_state:
    st.session_state.qa_system = None

if "available_domains" not in st.session_state:
    st.session_state.available_domains = []

if "concepts_cache" not in st.session_state:
    st.session_state.concepts_cache = {}

# Add processing flag
if "processing" not in st.session_state:
    st.session_state.processing = False

# Add tracking for pending queries
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Use to prevent message duplication
if "processed_queries" not in st.session_state:
    st.session_state.processed_queries = set()


def patch_get_concept_relationships():
    """
    Monkey patch the _get_concept_relationships method to fix the UNION query error.
    This fixes the "All sub queries in an UNION must have the same return column names" error.
    """
    if not st.session_state.qa_system:
        return

    # Define the fixed method
    def fixed_get_concept_relationships(self, concept_name, limit=10):
        """Get relationships for a concept."""
        try:
            with self.driver.session() as session:
                # Fixed query with matching column names in both parts of the UNION
                result = session.run("""
                MATCH (c:Concept {name: $name})-[r]->(related)
                RETURN type(r) AS relation, related.name AS related_concept, r.justification AS justification, 'outgoing' AS direction
                UNION
                MATCH (related)-[r]->(c:Concept {name: $name})
                RETURN type(r) AS relation, related.name AS related_concept, r.justification AS justification, 'incoming' AS direction
                LIMIT $limit
                """, {"name": concept_name, "limit": limit})

                relationships = []
                for record in result:
                    rel_dict = dict(record)
                    # Convert to the expected format based on direction
                    if rel_dict["direction"] == "outgoing":
                        rel_dict["target"] = rel_dict["related_concept"]
                    else:
                        rel_dict["source"] = rel_dict["related_concept"]
                    # Remove the temporary field
                    del rel_dict["related_concept"]
                    relationships.append(rel_dict)

                return relationships
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error getting relationships: {e}")
            return []

    # Apply the patch
    if hasattr(st.session_state.qa_system, '_get_concept_relationships'):
        st.session_state.qa_system._get_concept_relationships = fixed_get_concept_relationships.__get__(
            st.session_state.qa_system, type(st.session_state.qa_system)
        )


def initialize_qa_system():
    """Initialize the QA system and connect to Neo4j"""
    if st.session_state.qa_system is not None:
        # Close existing connection
        try:
            st.session_state.qa_system.close()
        except:
            pass

    with st.spinner("Connecting to knowledge graph..."):
        try:
            qa = EnhancedKnowledgeGraphQA(
                neo4j_uri=NEO4J_URI,
                neo4j_user=NEO4J_USER,
                neo4j_password=NEO4J_PASSWORD,
                openai_api_key=OPENAI_API_KEY,
                model_name=OPENAI_MODEL
            )
            st.session_state.qa_system = qa

            # Load available domains
            load_available_domains(qa)

            # Apply patch to fix Neo4j query if you're still using it
            patch_get_concept_relationships()

            return True
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {str(e)}")
            return False

def load_available_domains(qa):
    """Load available domains from the knowledge graph"""
    try:
        with qa.driver.session() as session:
            result = session.run("""
            MATCH (c:Concept)
            WHERE c.domain IS NOT NULL AND c.domain <> ''
            WITH DISTINCT c.domain AS domain
            RETURN collect(domain) AS domains
            """)
            record = result.single()
            if record:
                st.session_state.available_domains = sorted(record["domains"])
    except Exception as e:
        st.warning(f"Failed to load domains: {str(e)}")
        st.session_state.available_domains = []


def search_concepts(query, limit=5):
    """Search for concepts matching the query"""
    if not st.session_state.qa_system:
        return []

    if query in st.session_state.concepts_cache:
        return st.session_state.concepts_cache[query]

    try:
        concepts = st.session_state.qa_system._search_concepts(query, limit=limit)
        st.session_state.concepts_cache[query] = concepts
        return concepts
    except Exception as e:
        st.warning(f"Failed to search concepts: {str(e)}")
        return []


def format_analogy(analogy):
    """Format an analogy for display"""
    if isinstance(analogy, dict):
        source = analogy.get('source_concept', 'Unknown')
        target = analogy.get('target_concept', 'Unknown')
        explanation = analogy.get('explanation', '')
        score = analogy.get('similarity_score', 0)
        patterns = analogy.get('common_patterns', [])
    else:
        # Handle AnalogyConcept object
        source = analogy.source_concept
        target = analogy.target_concept
        explanation = analogy.explanation
        score = analogy.similarity_score
        patterns = analogy.common_patterns

    formatted = f"### {source} ↔ {target} (Similarity: {score:.2f})\n\n"
    formatted += f"{explanation}\n\n"

    if patterns:
        formatted += "**Common patterns:**\n"
        for pattern in patterns:
            formatted += f"- {pattern}\n"

    return formatted


def format_idea(idea):
    """Format an idea for display"""
    if isinstance(idea, dict):
        title = idea.get('title', 'Untitled Idea')
        description = idea.get('description', '')
        sources = idea.get('source_concepts', [])
        domains = idea.get('domains_connected', [])
        applications = idea.get('potential_applications', [])
        score = idea.get('novelty_score', 0)
    else:
        # Handle CrossDomainIdea object
        title = idea.title
        description = idea.description
        sources = idea.source_concepts
        domains = idea.domains_connected
        applications = idea.potential_applications
        score = idea.novelty_score

    formatted = f"### {title} (Novelty: {score:.2f})\n\n"
    formatted += f"{description}\n\n"

    if sources:
        formatted += f"**Source concepts:** {', '.join(sources)}\n\n"

    if domains:
        formatted += f"**Domains connected:** {', '.join(domains)}\n\n"

    if applications:
        formatted += "**Potential applications:**\n"
        for app in applications:
            formatted += f"- {app}\n"

    return formatted


def add_user_message(user_query):
    """
    First step: Add the user message to chat history and set up for processing.
    This ensures the user message is displayed immediately.
    """
    # Skip if we've already processed this exact query
    query_hash = hash(user_query)
    if query_hash in st.session_state.processed_queries:
        return

    # Add to processed set to prevent duplicates
    st.session_state.processed_queries.add(query_hash)

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Mark as processing and store query for processing on next rerun
    st.session_state.processing = True
    st.session_state.pending_query = user_query

    # Force a rerun to show the message immediately
    st.rerun()


def process_pending_query():
    """
    Second step: Process the pending query if one exists.
    This runs after the user message has been displayed.
    """
    if not st.session_state.pending_query or not st.session_state.processing:
        return

    user_query = st.session_state.pending_query

    # Create a placeholder for the assistant message
    message_placeholder = st.chat_message("assistant").empty()
    message_placeholder.markdown("Thinking...")

    # Process the query
    try:
        # Get the query type and creativity
        query_type = st.session_state.query_type
        creativity = getattr(st.session_state, "creativity", 0.7)

        # Create a structured query
        query_obj = KnowledgeGraphQuery(
            query=user_query,
            query_type=QueryType(query_type),
            creativity=creativity
        )

        # Add selected domains if specified
        selected_domains = getattr(st.session_state, "selected_domains", None)
        if selected_domains:
            query_obj.domains = selected_domains

        # Process the query
        start_time = time.time()
        result = st.session_state.qa_system.process_query(query_obj)
        elapsed_time = time.time() - start_time

        # For visualizations that need to be rendered in the UI
        visualization_container = None

        # Format initial response based on query type
        if query_type == "factual":
            response = result.get("answer", "I couldn't find an answer to that question.")
            # Add sources if available
            if "sources" in result and result["sources"]:
                sources_text = "\n\n**Sources:** " + ", ".join(result["sources"])
                response += sources_text
        else:
            # For non-factual queries, we'll use a more interactive display
            response = f"I've analyzed your query about '{user_query}' and prepared an interactive response."
            # Create a container for visualization that will be rendered after the message is displayed
            visualization_container = st.container()

        # Add processing metadata to text response
        meta = result.get("metadata", {})
        if meta:
            elapsed_msg = f"\n\n*Query processed in {elapsed_time:.2f} seconds"
            if "concepts_retrieved" in meta:
                elapsed_msg += f", using {meta.get('concepts_retrieved', 0)} concepts"
            if "paths_analyzed" in meta:
                elapsed_msg += f", analyzing {meta.get('paths_analyzed', 0)} knowledge paths"
            elapsed_msg += "*"
            response += elapsed_msg

        # Update the message placeholder with the text response
        message_placeholder.markdown(response)

        # Add to message history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # If we have a visualization container, render the enhanced visualization
        if visualization_container and query_type != "factual":
            with visualization_container:
                render_enhanced_query_results(result)

    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        message_placeholder.markdown(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})

    # Reset processing state
    st.session_state.processing = False
    st.session_state.pending_query = None


# UI Components
def render_sidebar():
    """Render the sidebar with configuration options"""
    with st.sidebar:
        st.title("Knowledge Graph QA")

        # Connection status
        if st.session_state.qa_system:
            st.success("Connected to knowledge graph")
        else:
            st.error("Not connected")
            if st.button("Connect"):
                initialize_qa_system()

        st.divider()

        # Query type selection
        st.subheader("Query Settings")
        st.selectbox(
            "Query Type",
            options=["factual", "analogy", "idea_generation", "comparison", "exploration"],
            index=0,
            help="Select the type of query to run",
            key="query_type"
        )

        # Creativity slider for idea generation
        if st.session_state.query_type == "idea_generation":
            st.slider(
                "Creativity Level",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher values produce more creative but potentially less practical ideas",
                key="creativity"
            )
        else:
            # Set default creativity
            st.session_state.creativity = 0.7

        # Domain selection
        if st.session_state.available_domains:
            st.multiselect(
                "Filter by Domains",
                options=st.session_state.available_domains,
                help="Select specific domains to focus on (optional)",
                key="selected_domains"
            )

        st.divider()

        # Quick actions
        st.subheader("Quick Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.session_state.processing = False
                st.session_state.pending_query = None
                st.session_state.processed_queries = set()

        with col2:
            if st.button("Reconnect"):
                initialize_qa_system()


def handle_concept_button(concept_name, action_type):
    """Handle concept button clicks by creating appropriate queries"""
    if action_type == "ask":
        query = f"What is {concept_name} and how is it used?"
    elif action_type == "analogy":
        st.session_state.query_type = "analogy"
        query = f"What are some analogies for {concept_name}?"
    elif action_type == "explore":
        st.session_state.query_type = "exploration"
        query = f"Explore the concept of {concept_name}"
    else:
        return

    add_user_message(query)


def render_chat():
    """Render the chat interface"""
    # Display chat header
    st.title("Knowledge Graph Explorer")
    st.write("Ask questions, find analogies, and generate ideas from your knowledge graph")

    # Display previous chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process pending query if one exists
    if st.session_state.pending_query and st.session_state.processing:
        process_pending_query()

    # Get user input
    input_disabled = st.session_state.processing
    user_query = st.chat_input(
        "What would you like to know about your knowledge graph?",
        disabled=input_disabled
    )

    if user_query and not input_disabled:
        add_user_message(user_query)


# Concept explorer component
def render_concept_explorer():
    """Render the concept explorer component"""
    with st.expander("Concept Explorer", expanded=False):
        st.write("Search for concepts in your knowledge graph")

        search_disabled = st.session_state.processing

        # Create a form for the search to ensure it triggers on submit
        with st.form(key="concept_search_form"):
            search_term = st.text_input("Search concepts", key="concept_search", disabled=search_disabled)
            submit_search = st.form_submit_button("Search", disabled=search_disabled)

        # Only process the search if form is submitted
        if submit_search and search_term and not search_disabled:
            concepts = search_concepts(search_term)

            if concepts:
                st.write(f"Found {len(concepts)} concepts:")

                for idx, concept in enumerate(concepts):
                    with st.container():
                        st.markdown(f"### {concept['name']}")
                        st.markdown(f"{concept['description']}")

                        # Add a button to create a query about this concept
                        col1, col2, col3 = st.columns(3)

                        button_disabled = st.session_state.processing

                        with col1:
                            if st.button(f"Ask about {concept['name']}", key=f"ask_{idx}", disabled=button_disabled):
                                handle_concept_button(concept['name'], "ask")

                        with col2:
                            if st.button(f"Find analogies for {concept['name']}", key=f"analogy_{idx}",
                                         disabled=button_disabled):
                                handle_concept_button(concept['name'], "analogy")

                        with col3:
                            if st.button(f"Explore {concept['name']}", key=f"explore_{idx}", disabled=button_disabled):
                                handle_concept_button(concept['name'], "explore")

                        st.divider()
            else:
                st.write("No concepts found matching your search")


# Main app
def main():
    """Main application function"""
    # Initialize connection to Neo4j if not already done
    if st.session_state.qa_system is None:
        initialize_qa_system()

    # Render sidebar
    render_sidebar()

    # Main content area
    render_chat()

    # Concept explorer at the bottom
    render_concept_explorer()


if __name__ == "__main__":
    main()