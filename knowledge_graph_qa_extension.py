# knowledge_graph_qa_extension.py
from knowledge_graph_qa import KnowledgeGraphQA, KnowledgeGraphQuery, QueryType
from knowledge_graph_traversal import KnowledgeGraphInsightEngine
import logging
import json
from typing import Dict, Any, List, Union, Optional

logger = logging.getLogger(__name__)


class EnhancedKnowledgeGraphQA(KnowledgeGraphQA):
    """
    Enhanced version of KnowledgeGraphQA with deep traversal capabilities.
    This extends the original class with path traversal for better insights.
    """
    
    def __init__(
            self,
            neo4j_uri: str,
            neo4j_user: str,
            neo4j_password: str,
            openai_api_key: Optional[str] = None,
            model_name: str = "gpt-4o-mini"
    ):
        # Initialize the parent class
        super().__init__(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            openai_api_key=openai_api_key,
            model_name=model_name
        )
        
        # Initialize the insight engine for deep traversal
        self.insight_engine = KnowledgeGraphInsightEngine(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            openai_api_key=openai_api_key,
            model_name=model_name
        )
        
        logger.info("Enhanced Knowledge Graph QA initialized with traversal capabilities")
    
    def generate_deep_insights(self, query_obj: KnowledgeGraphQuery) -> Dict[str, Any]:
        """
        Generate insights using deep graph traversal
        
        Args:
            query_obj: Knowledge graph query object
            
        Returns:
            Response dictionary with results and metadata
        """
        query_type = query_obj.query_type.value
        
        # Set appropriate depth based on query type
        if query_type == "idea_generation":
            max_depth = 5  # Allow deeper traversal for idea generation
            max_paths = 25
        elif query_type == "analogy":
            max_depth = 4
            max_paths = 20
        else:
            max_depth = 3
            max_paths = 15
            
        # Get domains if specified
        domains = query_obj.domains if hasattr(query_obj, 'domains') and query_obj.domains else None
            
        # Call the insight engine
        logger.info(f"Generating deep insights for query type: {query_type}")
        results = self.insight_engine.generate_insights(
            query=query_obj.query,
            query_type=query_type,
            max_concepts=query_obj.concept_limit,
            max_paths=max_paths,
            max_depth=max_depth,
            domains=domains
        )
        
        # Format response based on query type
        if query_type == "idea_generation":
            return {
                "ideas": results.get("ideas", []),
                "type": "idea_generation",
                "paths": results.get("paths", []),
                "concept_clusters": results.get("concept_clusters", []),
                "metadata": {
                    "concepts_retrieved": results.get("concepts_used", 0),
                    "paths_analyzed": results.get("paths_analyzed", 0),
                    "clusters_identified": len(results.get("concept_clusters", [])),
                    "query_type": query_type,
                    "query": query_obj.query
                }
            }
        elif query_type == "analogy":
            return {
                "analogies": results.get("analogies", []),
                "type": "analogy",
                "paths": results.get("paths", []),
                "metadata": {
                    "concepts_retrieved": results.get("concepts_used", 0),
                    "paths_analyzed": results.get("paths_analyzed", 0),
                    "query_type": query_type,
                    "query": query_obj.query
                }
            }
        elif query_type == "exploration":
            return {
                "exploration": results.get("response", {}).get("exploration", ""),
                "insights": results.get("response", {}).get("insights", []),
                "themes": results.get("response", {}).get("themes", []),
                "type": "exploration",
                "paths": results.get("paths", []),
                "concept_clusters": results.get("concept_clusters", []),
                "metadata": {
                    "concepts_retrieved": results.get("concepts_used", 0),
                    "paths_analyzed": results.get("paths_analyzed", 0),
                    "query_type": query_type,
                    "query": query_obj.query
                }
            }
        else:
            # For other types, pass through the response
            return {
                "response": results.get("response", {}),
                "type": query_type,
                "paths": results.get("paths", []),
                "metadata": {
                    "concepts_retrieved": results.get("concepts_used", 0),
                    "paths_analyzed": results.get("paths_analyzed", 0),
                    "query_type": query_type,
                    "query": query_obj.query
                }
            }
    
    def process_query(self, query: Union[str, KnowledgeGraphQuery]) -> Dict[str, Any]:
        """
        Process a user query with enhanced traversal capabilities.
        This overrides the parent class method to use deep traversal for appropriate queries.
        
        Args:
            query: Either a query string or a KnowledgeGraphQuery object
            
        Returns:
            Response dictionary with results
        """
        # Convert string to KnowledgeGraphQuery if needed
        if isinstance(query, str):
            # Classify the query type
            try:
                query_type_str = self.query_classifier_chain.invoke({"query": query})
                query_type = QueryType(query_type_str)
            except Exception as e:
                logger.warning(f"Error classifying query: {e}, defaulting to factual")
                # Default to factual if classification fails
                query_type = QueryType.FACTUAL

            # Create query object
            query_obj = KnowledgeGraphQuery(
                query=query,
                query_type=query_type
            )
        else:
            query_obj = query
            
        # For idea generation, analogy, and exploration queries, use deep traversal
        deep_traversal_types = [
            QueryType.IDEA_GENERATION, 
            QueryType.ANALOGY,
            QueryType.EXPLORATION
        ]
        
        if query_obj.query_type in deep_traversal_types:
            logger.info(f"Using deep traversal for query type: {query_obj.query_type}")
            return self.generate_deep_insights(query_obj)
        else:
            # Use parent class implementation for factual and comparison queries
            logger.info(f"Using standard processing for query type: {query_obj.query_type}")
            return super().process_query(query_obj)
    
    def close(self):
        """Close all database connections"""
        # Close the parent class connection
        super().close()
        
        # Close the insight engine connection
        if hasattr(self, 'insight_engine'):
            self.insight_engine.close()
