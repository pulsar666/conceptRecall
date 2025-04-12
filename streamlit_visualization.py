# streamlit_visualization.py
import streamlit as st
import json
import re
from typing import List, Dict, Any
# New imports for interactive visualization
import importlib.util
import tempfile
from pathlib import Path

def render_knowledge_paths(paths):
    """
    Render knowledge paths as a visual graph with a stable zoom approach.

    Args:
        paths: List of path dictionaries from the traversal engine
    """
    if not paths:
        st.info("No knowledge paths available to visualize")
        return

    try:
        # Use Streamlit's graphviz integration
        import graphviz

        # Initialize zoom level in session state if not present
        if "graph_zoom_level" not in st.session_state:
            st.session_state.graph_zoom_level = 1.0

        # Get current zoom level
        zoom = st.session_state.graph_zoom_level

        # Create a graph with settings adjusted for the current zoom level
        graph = graphviz.Digraph(engine='fdp')

        # Configure node appearance - adjust size based on zoom
        graph.attr('node',
                   shape='box',
                   style='filled,rounded',
                   fillcolor='lightblue',
                   fontsize=str(10 * zoom),  # Scale font with zoom
                   width=str(0.6 * zoom),  # Scale width with zoom
                   height=str(0.4 * zoom),  # Scale height with zoom
                   margin='0.1,0.1')

        # Configure overall graph
        graph.attr(rankdir='LR',
                   size=f'{6 * zoom},{4 * zoom}',  # Scale overall size with zoom
                   ratio='compress',
                   overlap='false',
                   splines='ortho',
                   fontsize=str(10 * zoom))  # Scale font with zoom

        # Add nodes and edges from paths
        added_nodes = set()
        added_edges = set()

        for path in paths:
            steps = path.get("steps", [])

            for step in steps:
                if not (step.get("source") and step.get("target") and step.get("relation")):
                    continue

                source = step["source"]
                target = step["target"]
                relation = step["relation"]

                # Add nodes if not already added
                if source not in added_nodes:
                    source_label = source if len(source) < 15 else source[:12] + "..."
                    graph.node(source, source_label)
                    added_nodes.add(source)
                if target not in added_nodes:
                    target_label = target if len(target) < 15 else target[:12] + "..."
                    graph.node(target, target_label)
                    added_nodes.add(target)

                # Add edge if not already added
                edge_key = f"{source}|{relation}|{target}"
                if edge_key not in added_edges:
                    rel_label = relation.replace('_', ' ').title()
                    rel_label = rel_label if len(rel_label) < 10 else rel_label[:7] + "..."
                    graph.edge(source, target, label=rel_label, fontsize=str(8 * zoom))
                    added_edges.add(edge_key)

        # Add title
        st.markdown("### Knowledge Graph Visualization")

        # Add zoom controls that don't trigger a full rerun
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if st.button("Zoom Out -"):
                st.session_state.graph_zoom_level = max(0.5, st.session_state.graph_zoom_level - 0.1)

        with col2:
            # Use a slider with key that doesn't change the session state directly
            # Only update on release to prevent disappearing during drag
            zoom_value = st.slider(
                "Zoom Level",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.graph_zoom_level,
                step=0.1,
                key="zoom_slider_display"  # Different key than session state
            )
            # Only update if value has changed
            if zoom_value != st.session_state.graph_zoom_level:
                st.session_state.graph_zoom_level = zoom_value

        with col3:
            if st.button("Zoom In +"):
                st.session_state.graph_zoom_level = min(2.0, st.session_state.graph_zoom_level + 0.1)

        # Render graph
        st.graphviz_chart(graph, use_container_width=True)

        st.caption(
            f"Knowledge graph visualization showing {len(added_nodes)} concepts and {len(added_edges)} relationships")

    except Exception as e:
        st.warning(f"Could not render knowledge paths: {str(e)}")


def render_knowledge_paths_interactive(paths):
    """
    Render knowledge paths with interactive zoom/pan using Pyvis.

    Args:
        paths: List of path dictionaries from the traversal engine
    """
    if not paths:
        st.info("No knowledge paths available to visualize")
        return

    try:
        # Only import these if needed
        from pyvis.network import Network
        import networkx as nx
        import tempfile
        from pathlib import Path
        import streamlit.components.v1 as components

        # Create a NetworkX graph
        G = nx.DiGraph()

        # Add nodes and edges from paths
        for path in paths:
            steps = path.get("steps", [])

            for step in steps:
                if not (step.get("source") and step.get("target") and step.get("relation")):
                    continue

                source = step["source"]
                target = step["target"]
                relation = step["relation"]

                # Add nodes if not already in graph
                if source not in G.nodes:
                    G.add_node(source, label=source, title=source)
                if target not in G.nodes:
                    G.add_node(target, label=target, title=target)

                # Add edge
                G.add_edge(source, target, label=relation, title=relation)

        # Create Pyvis network
        net = Network(height="450px", width="100%", directed=True, notebook=False)

        # Configure physics for better layout
        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -2000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "stabilization": {
              "enabled": true,
              "iterations": 1000
            }
          },
          "nodes": {
            "shape": "box",
            "color": {
              "background": "#D2E5FF",
              "border": "#2B7CE9"
            },
            "font": {
              "size": 12
            }
          },
          "edges": {
            "color": "#848484",
            "smooth": {
              "type": "continuous",
              "forceDirection": "none"
            },
            "font": {
              "size": 10,
              "align": "middle"
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.5
              }
            }
          },
          "interaction": {
            "navigationButtons": true,
            "keyboard": {
              "enabled": true
            }
          }
        }
        """)

        # Add the NetworkX graph to Pyvis
        net.from_nx(G)

        # Save and render HTML
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
            net.save_graph(tmpfile.name)

        # Load and display the HTML file
        with open(tmpfile.name, 'r', encoding='utf-8') as f:
            html_data = f.read()

        # Add a title
        st.markdown("### Interactive Knowledge Graph")

        # Display the HTML
        components.html(html_data, height=500, scrolling=False)

        # Clean up
        Path(tmpfile.name).unlink()

        st.caption(f"Knowledge graph visualization showing {len(G.nodes)} concepts and {len(G.edges)} relationships")
        st.caption(
            "You can zoom with the mouse wheel, pan by dragging the background, and move nodes by dragging them.")

    except Exception as e:
        st.error(f"Could not render interactive graph: {str(e)}")
        # Fall back to standard graphviz rendering
        render_knowledge_paths(paths)

def render_concept_clusters(clusters, max_concepts_per_cluster=8):
    """
    Render concept clusters as expandable sections.

    Args:
        clusters: List of cluster dictionaries
        max_concepts_per_cluster: Maximum concepts to show per cluster
    """
    if not clusters:
        st.info("No concept clusters available to visualize")
        return

    # Debug info
    st.write(f"Found {len(clusters)} concept clusters")

    for i, cluster in enumerate(clusters):
        coherence = cluster.get("coherence", 0)
        theme = cluster.get("theme", f"Cluster {i + 1}")
        concepts = cluster.get("concepts", [])

        if not concepts:
            continue

        with st.expander(f"{theme} (Coherence: {coherence:.2f})"):
            # Create a nice table-like display with cards
            total_shown = min(len(concepts), max_concepts_per_cluster)
            st.write(f"Showing {total_shown} of {len(concepts)} concepts in this cluster")

            # Use columns for a more compact display
            cols = st.columns(2)

            # Split concepts between columns
            half = total_shown // 2

            for j, concept in enumerate(concepts[:total_shown]):
                col_idx = 0 if j < half else 1

                with cols[col_idx]:
                    name = concept.get("name", "Unknown")
                    what = concept.get("what", "")
                    domain = concept.get("domain", "")

                    # Create a card-like container
                    with st.container():
                        st.markdown(f"**{name}**")
                        if what:
                            st.markdown(f"{what[:100]}..." if len(what) > 100 else what)
                        if domain:
                            st.caption(f"Domain: {domain}")
                        st.divider()

            # Show count if there are more concepts
            if len(concepts) > max_concepts_per_cluster:
                st.caption(f"... and {len(concepts) - max_concepts_per_cluster} more concepts")


def render_idea(idea):
    """
    Render a single idea in a structured way.
    
    Args:
        idea: Idea dictionary
    """
    if not idea:
        return
        
    title = idea.get("title", "Untitled Idea")
    description = idea.get("description", "")
    key_concepts = idea.get("key_concepts", []) or idea.get("source_concepts", [])
    applications = idea.get("applications", []) or idea.get("potential_applications", [])
    novelty = idea.get("novelty_score", 0)
    
    st.markdown(f"### {title}")
    st.progress(novelty, text=f"Novelty: {novelty:.2f}")
    
    st.markdown(description)
    
    if key_concepts:
        st.markdown("**Key concepts:**")
        # Wrap tags in a container for better styling
        with st.container():
            cols = st.columns(min(5, len(key_concepts)))
            for i, concept in enumerate(key_concepts):
                cols[i % 5].markdown(f"<div style='background-color: #e0f7fa; padding: 5px 10px; border-radius: 15px; margin: 5px 0; display: inline-block;'>{concept}</div>", unsafe_allow_html=True)
    
    if applications:
        st.markdown("**Potential applications:**")
        for app in applications:
            st.markdown(f"- {app}")


def render_analogy(analogy):
    """
    Render a single analogy in a structured way.
    
    Args:
        analogy: Analogy dictionary
    """
    if not analogy:
        return
        
    source = analogy.get("source_concept", "Unknown")
    target = analogy.get("target_concept", "Unknown")
    explanation = analogy.get("explanation", "")
    patterns = analogy.get("common_patterns", [])
    similarity = analogy.get("similarity_score", 0)
    insight = analogy.get("insight", "")
    
    st.markdown(f"### {source} ↔ {target}")
    st.progress(similarity, text=f"Similarity: {similarity:.2f}")
    
    st.markdown(explanation)
    
    if insight:
        st.markdown(f"**Key insight:** {insight}")
    
    if patterns:
        st.markdown("**Common patterns:**")
        for pattern in patterns:
            st.markdown(f"- {pattern}")


def render_enhanced_query_results(result):
    """
    Render the enhanced query results with path visualization.

    Args:
        result: Result dictionary from the enhanced knowledge graph QA
    """
    query_type = result.get("type", "factual")

    # First, create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Knowledge Paths", "Concept Clusters", "Analysis Results"])

    # Tab 1: Path visualization
    with tab1:
        if "paths" in result and result["paths"]:
            # Try to use the interactive visualization first
            try:
                # Check if pyvis is available
                import importlib
                if importlib.util.find_spec("pyvis") and importlib.util.find_spec("networkx"):
                    render_knowledge_paths_interactive(result["paths"])
                else:
                    # Fall back to standard visualization if pyvis is not installed
                    st.info("For a more interactive visualization, install pyvis: pip install pyvis networkx")
                    render_knowledge_paths(result["paths"])
            except Exception as e:
                # Fall back to standard visualization if there's an error
                st.warning(f"Interactive visualization failed: {str(e)}")
                render_knowledge_paths(result["paths"])
        else:
            st.info("No knowledge paths available")

    # Tab 2: Concept clusters
    with tab2:
        if "concept_clusters" in result and result["concept_clusters"]:
            render_concept_clusters(result["concept_clusters"])
        else:
            st.info("No concept clusters available")

    # Tab 3: Results based on query type
    with tab3:
        if query_type == "idea_generation":
            ideas = result.get("ideas", [])
            if ideas:
                st.markdown("## Generated Ideas")
                for idea in ideas:
                    render_idea(idea)
                    st.divider()
            else:
                st.info("No ideas generated")

        elif query_type == "analogy":
            analogies = result.get("analogies", [])
            if analogies:
                st.markdown("## Generated Analogies")
                for analogy in analogies:
                    render_analogy(analogy)
                    st.divider()
            else:
                st.info("No analogies generated")

        elif query_type == "exploration":
            exploration = result.get("exploration", "")
            if exploration:
                st.markdown("## Knowledge Space Exploration")
                st.markdown(exploration)

                # Show themes if available
                themes = result.get("themes", [])
                if themes:
                    st.markdown("### Key Themes")
                    for theme in themes:
                        st.markdown(f"- {theme}")

                # Show insights if available
                insights = result.get("insights", [])
                if insights:
                    st.markdown("### Key Insights")
                    for insight in insights:
                        st.markdown(f"- {insight}")
            else:
                st.info("No exploration results available")

    # Show metadata in a collapsible section at the bottom
    metadata = result.get("metadata", {})
    if metadata:
        with st.expander("Query Metadata", expanded=False):
            st.json(metadata)