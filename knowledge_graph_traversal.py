# knowledge_graph_traversal.py
import logging
import json
import time
from typing import List, Dict, Any, Optional, Union, Set
from neo4j import GraphDatabase
import openai
from knowledge_graph_path_scoring import PathScoringEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphTraversalEngine:
    """Engine for traversing paths in the knowledge graph"""
    
    def __init__(self, neo4j_driver, max_depth=4, max_paths=10):
        self.driver = neo4j_driver
        self.max_depth = max_depth
        self.max_paths = max_paths
        
    def traverse_paths(self, seed_concepts, traversal_strategy="breadth_first", path_filters=None):
        """
        Traverse paths starting from seed concepts using the specified strategy.
        
        Args:
            seed_concepts: List of starting concept names
            traversal_strategy: "breadth_first", "depth_first", or "bidirectional"
            path_filters: Optional filters to apply during traversal (domains, relation types)
            
        Returns:
            List of paths, each containing sequence of concepts and relationships
        """
        if not seed_concepts:
            logger.warning("No seed concepts provided for traversal")
            return []
            
        logger.info(f"Starting traversal with strategy: {traversal_strategy}, seeds: {seed_concepts}")
        
        if traversal_strategy == "breadth_first":
            return self._breadth_first_traverse(seed_concepts, path_filters)
        elif traversal_strategy == "bidirectional":
            return self._bidirectional_traverse(seed_concepts, path_filters)
        else:
            return self._depth_first_traverse(seed_concepts, path_filters)

    def _breadth_first_traverse(self, seed_concepts, path_filters):
        """Implementation of breadth-first traversal with optimized Cypher queries"""
        paths = []

        with self.driver.session() as session:
            # Use string concatenation for the variable path length
            # This is the correct way to use dynamic path lengths in Neo4j
            depth = str(self.max_depth)
            query = """
            MATCH path = (start:Concept)-[r*1..""" + depth + """]-(end:Concept)
            WHERE start.name IN $seed_concepts
            AND start <> end
            """

            # Add domain filters if specified
            if path_filters and 'domains' in path_filters:
                query += """
                AND ALL(n IN nodes(path) WHERE n.domain IN $domains OR n.name IN $seed_concepts)
                """

            # Add relationship filters if specified
            if path_filters and 'relation_types' in path_filters:
                query += """
                AND ALL(rel IN relationships(path) WHERE type(rel) IN $relation_types)
                """

            query += """
            RETURN path
            LIMIT $max_paths
            """

            params = {
                "seed_concepts": seed_concepts,
                "max_paths": self.max_paths
            }

            # Add filter parameters if needed
            if path_filters:
                if 'domains' in path_filters:
                    params["domains"] = path_filters['domains']
                if 'relation_types' in path_filters:
                    params["relation_types"] = path_filters['relation_types']

            try:
                # Execute query with parameters
                result = session.run(query, params)

                # Process results into path objects
                for record in result:
                    path = record["path"]
                    path_data = self._format_path(path)
                    paths.append(path_data)

                logger.info(f"Breadth-first traversal found {len(paths)} paths")
                return paths

            except Exception as e:
                logger.error(f"Error in breadth-first traversal: {str(e)}")
                # Try fallback with simpler query and fixed length
                try:
                    # Use a fixed length for the fallback
                    simplified_query = """
                    MATCH path = (start:Concept)-[r*1..3]-(end:Concept)
                    WHERE start.name IN $seed_concepts
                    RETURN path
                    LIMIT $max_paths
                    """

                    result = session.run(simplified_query, {
                        "seed_concepts": seed_concepts,
                        "max_paths": self.max_paths
                    })

                    for record in result:
                        path = record["path"]
                        path_data = self._format_path(path)
                        paths.append(path_data)

                    logger.info(f"Fallback traversal found {len(paths)} paths")
                    return paths

                except Exception as e2:
                    logger.error(f"Fallback traversal also failed: {str(e2)}")
                    return []
        
    def _bidirectional_traverse(self, seed_concepts, path_filters):
        """
        Bidirectional traversal between pairs of seed concepts.
        More efficient for finding connections between specific concepts.
        """
        if len(seed_concepts) < 2:
            return self._breadth_first_traverse(seed_concepts, path_filters)
            
        paths = []
        # Create pairs of concepts for bidirectional search
        concept_pairs = [(seed_concepts[i], seed_concepts[j]) 
                        for i in range(len(seed_concepts)) 
                        for j in range(i+1, len(seed_concepts))]

        with self.driver.session() as session:
            for source, target in concept_pairs:
                query = """
                   MATCH path = shortestPath((source:Concept {name: $source})-[*1..$max_depth]-(target:Concept {name: $target}))
                   """
                # Add domain filters if specified
                if path_filters and 'domains' in path_filters:
                    query += """
                    WHERE ALL(n IN nodes(path) WHERE n.domain IN $domains OR n.name IN $seed_concepts)
                    """
                    
                query += "RETURN path"
                
                params = {
                    "source": source,
                    "target": target,
                    "max_depth": self.max_depth
                }
                
                # Add filter parameters if needed
                if path_filters and 'domains' in path_filters:
                    params["domains"] = path_filters['domains']
                    params["seed_concepts"] = seed_concepts
                
                try:
                    result = session.run(query, params)
                    
                    for record in result:
                        path = record["path"]
                        path_data = self._format_path(path)
                        paths.append(path_data)
                except Exception as e:
                    logger.error(f"Error in bidirectional search between {source} and {target}: {str(e)}")
                    # Continue with next pair
        
        logger.info(f"Bidirectional traversal found {len(paths)} paths between {len(concept_pairs)} concept pairs")
        return paths

    def _depth_first_traverse(self, seed_concepts, path_filters):
        """Implementation of depth-first traversal"""
        # For Neo4j, the traversal is handled by the database, so the implementation
        # is similar to breadth-first but with different path selection criteria

        paths = []

        with self.driver.session() as session:
            # Use a slightly different query to prioritize deeper paths
            query = """
            MATCH path = (start:Concept)-[r*1..$max_depth]->(end:Concept)
            WHERE start.name IN $seed_concepts
            AND start <> end
            """

            # Add filters similar to breadth-first
            if path_filters and 'domains' in path_filters:
                query += """
                AND ALL(n IN nodes(path) WHERE n.domain IN $domains OR n.name IN $seed_concepts)
                """

            # Prioritize longer paths in depth-first search
            query += """
            WITH path, length(path) AS path_length
            ORDER BY path_length DESC
            RETURN path
            LIMIT $max_paths
            """
            
            params = {
                "seed_concepts": seed_concepts,
                "max_depth": self.max_depth,
                "max_paths": self.max_paths
            }
            
            if path_filters and 'domains' in path_filters:
                params["domains"] = path_filters['domains']
            
            try:
                result = session.run(query, params)
                
                for record in result:
                    path = record["path"]
                    path_data = self._format_path(path)
                    paths.append(path_data)
                    
            except Exception as e:
                logger.error(f"Error in depth-first traversal: {str(e)}")
                # No fallback needed as breadth-first already has one
        
        logger.info(f"Depth-first traversal found {len(paths)} paths")
        return paths
    
    def _format_path(self, path):
        """Convert Neo4j path to structured data"""
        try:
            nodes = path.nodes
            relationships = path.relationships
            
            formatted_path = {
                "concepts": [],
                "relationships": [],
                "steps": []
            }
            
            # Extract nodes
            for node in nodes:
                node_data = dict(node.items())
                concept = {
                    "name": node_data.get("name", "Unknown"),
                    "what": node_data.get("what", ""),
                    "domain": node_data.get("domain", "")
                }
                formatted_path["concepts"].append(concept)
            
            # Extract relationships and build steps
            for i, rel in enumerate(relationships):
                rel_data = {
                    "type": rel.type,
                    "justification": dict(rel.items()).get("justification", "")
                }
                formatted_path["relationships"].append(rel_data)
                
                # Create step (source -> relation -> target)
                source_idx = rel.start_node.id
                target_idx = rel.end_node.id
                
                # Find node indices
                source_node = None
                target_node = None
                for j, node in enumerate(nodes):
                    if node.id == source_idx:
                        source_node = node
                    if node.id == target_idx:
                        target_node = node
                    if source_node and target_node:
                        break
                
                if source_node and target_node:
                    step = {
                        "source": source_node["name"],
                        "relation": rel.type,
                        "target": target_node["name"]
                    }
                    formatted_path["steps"].append(step)
            
            return formatted_path
            
        except Exception as e:
            logger.error(f"Error formatting path: {str(e)}")
            # Return minimal valid path structure
            return {
                "concepts": [],
                "relationships": [],
                "steps": []
            }


class ConceptClusterEngine:
    """Engine for clustering concepts from the knowledge graph"""
    
    def __init__(self, neo4j_driver, openai_client, model="text-embedding-3-small"):
        self.driver = neo4j_driver
        self.client = openai_client
        self.model = model
        self.embedding_cache = {}
        
    def find_concept_clusters(self, concepts, min_concepts_per_cluster=3, max_clusters=5):
        """
        Find meaningful clusters of concepts based on embeddings
        
        Args:
            concepts: List of concept dictionaries
            min_concepts_per_cluster: Minimum concepts to form a cluster
            max_clusters: Maximum number of clusters to return
            
        Returns:
            List of clusters, each with concept list and common theme
        """
        # Skip if not enough concepts
        if len(concepts) < min_concepts_per_cluster:
            return [{"concepts": concepts, "theme": "Main Cluster", "coherence": 1.0}]
            
        # Extract embeddings
        concept_embeddings = []
        valid_concepts = []
        
        for concept in concepts:
            name = concept["name"]
            # Get embedding from concept if available, otherwise generate new one
            if name in self.embedding_cache:
                embedding = self.embedding_cache[name]
                concept_embeddings.append(embedding)
                valid_concepts.append(concept)
            elif 'embedding' in concept and concept['embedding']:
                embedding = concept['embedding']
                self.embedding_cache[name] = embedding
                concept_embeddings.append(embedding)
                valid_concepts.append(concept)
            else:
                # Generate new embedding
                embedding = self._generate_embedding(concept['name'] + ": " + concept.get('what', ''))
                if embedding:
                    self.embedding_cache[name] = embedding
                    concept_embeddings.append(embedding)
                    valid_concepts.append(concept)
        
        # If no valid concepts with embeddings, return a single cluster
        if not valid_concepts:
            return [{"concepts": concepts, "theme": "Main Cluster", "coherence": 1.0}]
            
        # Perform clustering using scikit-learn
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
            import numpy as np
            
            # Convert embeddings to numpy array
            X = np.array(concept_embeddings)
            
            # Determine optimal number of clusters (between 2 and max_clusters)
            max_k = min(max_clusters, len(valid_concepts) // min_concepts_per_cluster)
            max_k = max(2, max_k)  # At least 2 clusters if possible
            
            best_k = 2
            best_score = -1
            
            # Only attempt clustering if we have enough samples
            if len(valid_concepts) >= 2 * min_concepts_per_cluster:
                for k in range(2, max_k + 1):
                    clustering = AgglomerativeClustering(n_clusters=k).fit(X)
                    labels = clustering.labels_
                    
                    if len(np.unique(labels)) < 2:
                        continue
                        
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                        
                clustering = AgglomerativeClustering(n_clusters=best_k).fit(X)
                labels = clustering.labels_
            else:
                # Not enough samples, create a single cluster
                labels = np.zeros(len(valid_concepts))
                
            # Group concepts by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(valid_concepts[i])
                
            # Format results and identify themes
            result = []
            for label, cluster_concepts in clusters.items():
                # Skip too small clusters
                if len(cluster_concepts) < min_concepts_per_cluster:
                    continue
                    
                theme = self._identify_cluster_theme(cluster_concepts)
                result.append({
                    "concepts": cluster_concepts,
                    "theme": theme,
                    "coherence": self._calculate_coherence(cluster_concepts)
                })
                
            return result
            
        except Exception as e:
            logger.error(f"Error in concept clustering: {str(e)}")
            # Fallback: return single cluster if clustering fails
            return [{"concepts": concepts, "theme": "Main Group", "coherence": 1.0}]
            
    def _generate_embedding(self, text):
        """Generate embedding for text using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
            
    def _identify_cluster_theme(self, concepts):
        """Identify common theme for a cluster of concepts"""
        try:
            # Extract concept information
            concept_texts = []
            for concept in concepts[:10]:  # Limit to 10 concepts for API efficiency
                text = f"{concept['name']}: {concept.get('what', '')}"
                concept_texts.append(text)
                
            # Join texts with newlines
            concepts_text = "\n".join(concept_texts)
                
            # Send to OpenAI to identify theme
            prompt = f"""
            Identify the common theme that connects these concepts:
            
            {concepts_text}
            
            Return ONLY the theme name, nothing else.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error identifying cluster theme: {str(e)}")
            # Fallback theme based on most common domain
            domains = [c.get('domain', '') for c in concepts if 'domain' in c and c['domain']]
            if domains:
                from collections import Counter
                common_domain = Counter(domains).most_common(1)[0][0]
                return f"{common_domain} Concepts"
            return "Related Concepts"
            
    def _calculate_coherence(self, concepts):
        """Calculate coherence score for a cluster (0-1)"""
        # Simple implementation based on domain consistency
        domains = [c.get('domain', '') for c in concepts if 'domain' in c and c['domain']]
        if not domains:
            return 0.5
            
        from collections import Counter
        domain_counts = Counter(domains)
        most_common = domain_counts.most_common(1)[0][1]
        return most_common / len(domains)  # Higher score = more coherent (same domain)


class PathAnalysisEngine:
    """Engine for analyzing paths and extracting insights"""
    
    def __init__(self, openai_client, model="gpt-4o-mini"):
        self.client = openai_client
        self.model = model
        
    def analyze_paths(self, paths, analysis_type="insights"):
        """
        Analyze the provided paths to extract insights
        
        Args:
            paths: List of paths from GraphTraversalEngine
            analysis_type: Type of analysis to perform (insights, connections, patterns)
            
        Returns:
            Analysis results based on the requested type
        """
        if not paths:
            logger.warning("No paths provided for analysis")
            return self._empty_analysis_result(analysis_type)
            
        if analysis_type == "connections":
            return self._analyze_connections(paths)
        elif analysis_type == "patterns":
            return self._analyze_patterns(paths)
        else:
            return self._extract_insights(paths)
            
    def _extract_insights(self, paths):
        """Extract key insights from paths"""
        # Prepare path data for analysis
        path_texts = []
        
        # Limit to 10 paths to avoid token limits
        for i, path in enumerate(paths[:10]):
            path_text = f"Path {i+1}:\n"
            for step in path.get("steps", []):
                path_text += f"  {step.get('source', '')} → [{step.get('relation', '')}] → {step.get('target', '')}\n"
            path_texts.append(path_text)
            
        # Combine paths for analysis
        combined_paths = "\n".join(path_texts)
        
        # Get unique concepts across all paths
        unique_concepts = self._extract_unique_concepts(paths)
        
        # Send to LLM for insight extraction
        prompt = f"""
        Analyze these paths through a knowledge graph:
        
        {combined_paths}
        
        Extract the most significant insights, including:
        1. Non-obvious connections between distant concepts
        2. Emergent patterns or principles
        3. Potential novel applications or ideas
        4. Knowledge gaps or areas for further exploration
        
        Format your response as a JSON object:
        {{
          "key_insights": [
            {{
              "title": "Short insight title",
              "description": "Detailed explanation",
              "concepts_involved": ["concept1", "concept2"],
              "novelty_score": 0.75
            }}
          ],
          "emergent_themes": ["theme1", "theme2"],
          "knowledge_gaps": ["gap1", "gap2"]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            # Fallback simple response
            return {
                "key_insights": [
                    {
                        "title": "Path Analysis",
                        "description": f"Analyzed {len(paths)} paths between concepts",
                        "concepts_involved": unique_concepts[:5],
                        "novelty_score": 0.5
                    }
                ],
                "emergent_themes": [],
                "knowledge_gaps": []
            }
            
    def _analyze_connections(self, paths):
        """Analyze connections between concepts across paths"""
        # Extract all unique concepts from paths
        all_concepts = self._extract_unique_concepts(paths)
        
        # Extract all relationships between concepts
        relationships = {}
        for path in paths:
            for step in path.get("steps", []):
                source = step.get("source", "")
                target = step.get("target", "")
                relation = step.get("relation", "")
                
                if not (source and target and relation):
                    continue
                
                key = f"{source}_{target}"
                if key not in relationships:
                    relationships[key] = []
                    
                if relation not in relationships[key]:
                    relationships[key].append(relation)
        
        # Format for LLM analysis
        connection_text = ""
        for key, relations in list(relationships.items())[:20]:  # Limit to 20 for token limits
            source, target = key.split("_")
            relation_str = ", ".join(relations)
            connection_text += f"{source} → {relation_str} → {target}\n"
            
        # Send to LLM for analysis
        prompt = f"""
        Analyze these connections between concepts:
        
        {connection_text}
        
        Identify the most significant connections and potential implications.
        Focus on unexpected or high-value connections that might lead to novel insights.
        
        Format your response as a JSON object:
        {{
          "key_connections": [
            {{
              "source": "concept1",
              "target": "concept2",
              "significance": "Explanation of why this connection is significant",
              "potential_value": 0.85
            }}
          ],
          "connection_patterns": ["pattern1", "pattern2"],
          "suggested_new_connections": [
            {{
              "source": "concept1",
              "target": "concept3",
              "rationale": "Why these should be connected"
            }}
          ]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error analyzing connections: {str(e)}")
            # Fallback simple response
            return {
                "key_connections": [],
                "connection_patterns": [],
                "suggested_new_connections": []
            }
    
    def _analyze_patterns(self, paths):
        """Identify recurring patterns in relationship chains"""
        # Extract relationship patterns from paths
        patterns = []
        for path in paths:
            if "relationships" in path and len(path["relationships"]) >= 2:  # Need at least 2 relationships for a pattern
                pattern = []
                for rel in path["relationships"]:
                    pattern.append(rel.get("type", "UNKNOWN"))
                patterns.append(" → ".join(pattern))
            elif "steps" in path and len(path["steps"]) >= 2:
                pattern = []
                for step in path["steps"]:
                    pattern.append(step.get("relation", "UNKNOWN"))
                patterns.append(" → ".join(pattern))
                
        # Count pattern frequencies
        from collections import Counter
        pattern_counts = Counter(patterns)
        common_patterns = pattern_counts.most_common(5)
        
        # Format for analysis
        pattern_text = ""
        for pattern, count in common_patterns:
            pattern_text += f"{pattern}: {count} occurrences\n"
            
        # Send to LLM for pattern analysis
        prompt = f"""
        Analyze these relationship patterns found in knowledge paths:
        
        {pattern_text}
        
        Identify what these patterns might signify and how they could be leveraged to generate new insights.
        
        Format your response as a JSON object:
        {{
          "pattern_insights": [
            {{
              "pattern": "relation1 → relation2",
              "interpretation": "What this pattern typically represents",
              "implications": "How this pattern could be used"
            }}
          ],
          "meta_patterns": ["Higher-level pattern1", "Higher-level pattern2"],
          "suggested_applications": ["application1", "application2"]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            # Fallback simple response
            return {
                "pattern_insights": [],
                "meta_patterns": [],
                "suggested_applications": []
            }
    
    def _extract_unique_concepts(self, paths):
        """Extract all unique concepts from paths"""
        concepts = set()
        for path in paths:
            if "concepts" in path:
                for concept in path["concepts"]:
                    if isinstance(concept, dict) and "name" in concept:
                        concepts.add(concept["name"])
            elif "steps" in path:
                for step in path["steps"]:
                    if "source" in step:
                        concepts.add(step["source"])
                    if "target" in step:
                        concepts.add(step["target"])
        return list(concepts)
    
    def _empty_analysis_result(self, analysis_type):
        """Return empty analysis result structure based on type"""
        if analysis_type == "connections":
            return {
                "key_connections": [],
                "connection_patterns": [],
                "suggested_new_connections": []
            }
        elif analysis_type == "patterns":
            return {
                "pattern_insights": [],
                "meta_patterns": [],
                "suggested_applications": []
            }
        else:
            return {
                "key_insights": [],
                "emergent_themes": [],
                "knowledge_gaps": []
            }


class KnowledgeGraphInsightEngine:
    """Main engine for generating insights from knowledge graph traversal"""
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, openai_api_key, model_name="gpt-4o-mini"):
        """Initialize the insight engine with necessary components"""
        # Neo4j driver
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # OpenAI client
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model_name
        
        # Initialize component engines
        self.traversal_engine = GraphTraversalEngine(self.driver)
        self.cluster_engine = ConceptClusterEngine(self.driver, self.client)
        self.path_analysis_engine = PathAnalysisEngine(self.client, model=model_name)
        
        # Caching
        self.path_cache = {}
        self.concept_cache = {}

        self.path_scorer = PathScoringEngine(self.client)

    def generate_insights(self, query, query_type="idea_generation", max_concepts=15, max_paths=20, max_depth=4, domains=None):
        """
        Generate insights based on deep graph traversal
        
        Args:
            query: User query string or structured query object
            query_type: Type of insights to generate (idea_generation, analogies, patterns, exploration)
            max_concepts: Maximum concepts to use
            max_paths: Maximum paths to analyze
            max_depth: Maximum traversal depth
            domains: Optional list of domains to filter by
            
        Returns:
            Rich insights based on the knowledge graph structure
        """
        # Step 1: Extract seed concepts from query
        seed_concepts = self._extract_seed_concepts(query)
        
        # Step 2: Set traversal parameters based on query type
        self.traversal_engine.max_depth = max_depth
        self.traversal_engine.max_paths = max_paths
        
        # Step 3: Choose traversal strategy based on query type
        if query_type == "analogy":
            traversal_strategy = "bidirectional"
        elif query_type == "patterns":
            traversal_strategy = "depth_first"
        else:
            traversal_strategy = "breadth_first"
            
        # Step 4: Set up path filters
        path_filters = {}
        if domains:
            path_filters["domains"] = domains
            
        # Step 5: Traverse paths from seed concepts
        cache_key = f"{','.join(sorted(seed_concepts))}-{traversal_strategy}-{max_depth}"
        if cache_key in self.path_cache:
            paths = self.path_cache[cache_key]
        else:
            paths = self.traversal_engine.traverse_paths(
                seed_concepts, 
                traversal_strategy=traversal_strategy,
                path_filters=path_filters
            )
            self.path_cache[cache_key] = paths
            
        # Step 6: Extract all concepts from paths
        all_concepts = self._extract_concepts_from_paths(paths)
        
        # Step 7: Cluster concepts to find thematic groupings
        concept_clusters = self.cluster_engine.find_concept_clusters(all_concepts)

        # Step 8: Score paths and analyze top ones
        scored_paths = self.path_scorer.score_paths(paths, query)
        top_paths = [p["path"] for p in scored_paths[:10]]

        if query_type == "idea_generation":
            analysis = self.path_analysis_engine.analyze_paths(top_paths, analysis_type="insights")
        elif query_type == "analogy":
            analysis = self.path_analysis_engine.analyze_paths(top_paths, analysis_type="connections")
        elif query_type == "patterns":
            analysis = self.path_analysis_engine.analyze_paths(top_paths, analysis_type="patterns")
        else:  # exploration
            analysis = self.path_analysis_engine.analyze_paths(top_paths, analysis_type="insights")

        # Step 9: Generate response based on query type and analysis
        response = self._generate_response(query, query_type, concept_clusters, paths, analysis)

        # Return the combined results
        return {
            "query": query,
            "seed_concepts": seed_concepts,
            "paths": paths,
            "paths_analyzed": len(paths),
            "concepts_used": len(all_concepts),
            "concept_clusters": concept_clusters,
            "path_analysis": analysis,
            "response": response
        }

    def _extract_seed_concepts(self, query):
        """Extract seed concepts from query"""
        if isinstance(query, str):
            # Use LLM to extract concepts from query string
            prompt = f"""
            Extract the main concepts from this query:
            "{query}"

            List only the key nouns or subject terms that would be concepts in a knowledge graph.
            Return as a JSON array of strings, maximum 5 concepts.
            """

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )

                content = response.choices[0].message.content
                concepts = json.loads(content)

                # Check if we got a proper structure or just an array
                if isinstance(concepts, list):
                    return concepts[:5]
                elif isinstance(concepts, dict) and "concepts" in concepts:
                    return concepts["concepts"][:5]
                else:
                    # Try to extract from any field that might be a list
                    for key, value in concepts.items():
                        if isinstance(value, list) and len(value) > 0:
                            return value[:5]

                    # Fallback
                    return [query.split()[0]] if query.split() else ["knowledge"]

            except Exception as e:
                logger.error(f"Error extracting seed concepts: {str(e)}")
                # Simple fallback extraction
                words = query.split()
                return [w for w in words if len(w) > 3][:5]
        else:
            # Extract from structured query object
            if hasattr(query, "seed_concepts") and query.seed_concepts:
                return query.seed_concepts
            elif hasattr(query, "query"):
                return self._extract_seed_concepts(query.query)
            else:
                return []

    def _extract_concepts_from_paths(self, paths):
        """Extract all concepts from paths with deduplication"""
        concepts = {}
        for path in paths:
            if "concepts" in path:
                for concept in path["concepts"]:
                    if isinstance(concept, dict) and "name" in concept:
                        name = concept["name"]
                        if name not in concepts:
                            concepts[name] = concept
            elif "steps" in path:
                # If we only have steps, try to find concepts in Neo4j
                concept_names = set()
                for step in path["steps"]:
                    if "source" in step:
                        concept_names.add(step["source"])
                    if "target" in step:
                        concept_names.add(step["target"])

                # Look up these concepts
                for name in concept_names:
                    if name not in concepts:
                        concept_data = self._get_concept_by_name(name)
                        if concept_data:
                            concepts[name] = concept_data
                        else:
                            # Minimal concept data if not found
                            concepts[name] = {"name": name, "what": ""}

        return list(concepts.values())

    def _get_concept_by_name(self, name):
        """Get concept details from Neo4j"""
        if not name:
            return None

        try:
            with self.driver.session() as session:
                result = session.run("""
                MATCH (c:Concept {name: $name})
                RETURN c
                """, {"name": name})
                record = result.single()
                if not record:
                    return None

                node = record["c"]
                return {
                    "name": node.get("name"),
                    "aliases": node.get("aliases", []),
                    "what": node.get("what", ""),
                    "how": node.get("how", ""),
                    "why": node.get("why", ""),
                    "when": node.get("when", ""),
                    "domain": node.get("domain", ""),
                    "type": node.get("type", "")
                }
        except Exception as e:
            logger.error(f"Error getting concept by name: {e}")
            return None

    def _generate_response(self, query, query_type, concept_clusters, paths, analysis):
        """
        Generate final response based on query type and analysis

        Args:
            query: Original query string or object
            query_type: Type of query (idea_generation, analogy, exploration, etc.)
            concept_clusters: Discovered concept clusters
            paths: Knowledge paths traversed
            analysis: Results of path analysis

        Returns:
            Formatted response appropriate for the query type
        """
        """Generate final response based on query type and analysis"""
        # Debug logging
        logger.info(f"Generating response for query type: {query_type}")
        logger.info(f"Number of concept clusters: {len(concept_clusters)}")
        logger.info(f"Number of paths: {len(paths)}")
        if query_type == "idea_generation":
            # Format ideas from insights
            ideas = []
            if "key_insights" in analysis:
                for insight in analysis["key_insights"]:
                    ideas.append({
                        "title": insight.get("title", "Untitled Idea"),
                        "description": insight.get("description", ""),
                        "source_concepts": insight.get("concepts_involved", []),
                        "domains_connected": self._extract_domains_from_concepts(insight.get("concepts_involved", [])),
                        "potential_applications": self._generate_applications(insight),
                        "novelty_score": insight.get("novelty_score", 0.5)
                    })
            return {"ideas": ideas}

        elif query_type == "analogy":
            # Format analogies from connections
            analogies = []
            if "key_connections" in analysis:
                for connection in analysis["key_connections"]:
                    analogies.append({
                        "source_concept": connection.get("source", ""),
                        "target_concept": connection.get("target", ""),
                        "explanation": connection.get("significance", ""),
                        "common_patterns": self._extract_patterns_for_concepts(connection.get("source", ""),
                                                                               connection.get("target", ""),
                                                                               paths),
                        "similarity_score": connection.get("potential_value", 0.5)
                    })
            return {"analogies": analogies}

        elif query_type == "exploration":
            # Format exploration summary
            themes = []
            insights = []

            if "emergent_themes" in analysis:
                themes = analysis["emergent_themes"]

            if "key_insights" in analysis:
                insights = [insight.get("title", "") + ": " + insight.get("description", "")
                            for insight in analysis["key_insights"]]

            # Generate exploration summary
            exploration_text = self._generate_exploration_summary(query, concept_clusters, paths, analysis)

            return {
                "exploration": exploration_text,
                "themes": themes,
                "insights": insights
            }

        else:  # Default for other query types
            return {"analysis": analysis}

    def _extract_domains_from_concepts(self, concept_names):
        """Extract domains from a list of concept names"""
        domains = set()
        for name in concept_names:
            concept = self._get_concept_by_name(name)
            if concept and concept.get("domain"):
                domains.add(concept.get("domain"))
        return list(domains)

    def _generate_applications(self, insight):
        """Generate potential applications from an insight"""
        try:
            prompt = f"""
            Based on this insight:
            Title: {insight.get('title', '')}
            Description: {insight.get('description', '')}

            Generate 2-3 potential practical applications or use cases.
            List ONLY the applications as short phrases, one per line.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )

            applications = response.choices[0].message.content.strip().split("\n")
            return [app.strip() for app in applications if app.strip()]
        except Exception as e:
            logger.error(f"Error generating applications: {str(e)}")
            return ["Application based on this insight"]

    def _extract_patterns_for_concepts(self, concept1, concept2, paths):
        """Extract common patterns between two concepts from paths"""
        common_patterns = []

        # Look for paths containing both concepts
        for path in paths:
            concept_names = set()
            if "concepts" in path:
                for concept in path["concepts"]:
                    if isinstance(concept, dict) and "name" in concept:
                        concept_names.add(concept["name"])
            elif "steps" in path:
                for step in path["steps"]:
                    if "source" in step:
                        concept_names.add(step["source"])
                    if "target" in step:
                        concept_names.add(step["target"])

            # If both concepts are in this path, extract a pattern
            if concept1 in concept_names and concept2 in concept_names:
                pattern = self._extract_path_pattern(path)
                if pattern and pattern not in common_patterns:
                    common_patterns.append(pattern)

        if not common_patterns:
            common_patterns = ["Conceptual similarity"]

        return common_patterns[:3]  # Return up to 3 patterns

    def _extract_path_pattern(self, path):
        """Extract a meaningful pattern from a path"""
        if "relationships" in path and path["relationships"]:
            rel_types = [rel.get("type", "") for rel in path["relationships"] if "type" in rel]
            if rel_types:
                return " → ".join(rel_types)
        return None

    def _generate_exploration_summary(self, query, concept_clusters, paths, analysis):
        """Generate a summary for exploration queries"""
        try:
            # Extract key concepts for the summary
            concepts_text = ""
            for cluster in concept_clusters[:3]:  # Use top 3 clusters
                concepts_text += f"Cluster: {cluster.get('theme', 'Unnamed')}\n"
                for concept in cluster.get('concepts', [])[:5]:  # Top 5 concepts per cluster
                    concepts_text += f"- {concept.get('name', '')}: {concept.get('what', '')[:100]}\n"

            # Extract key insights
            insights_text = ""
            if "key_insights" in analysis:
                for insight in analysis["key_insights"]:
                    insights_text += f"- {insight.get('title', '')}: {insight.get('description', '')[:100]}\n"

            prompt = f"""
            Generate a concise exploration summary about the following query:
            "{query}"

            Based on these concept clusters:
            {concepts_text}

            And these key insights:
            {insights_text}

            Write 2-3 paragraphs that synthesize this information into a coherent exploration
            of the knowledge space. Focus on key themes, interesting connections, and potential
            areas for further exploration.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating exploration summary: {str(e)}")
            return f"Exploration of concepts related to the query: {query}"