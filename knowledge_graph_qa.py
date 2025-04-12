import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    FACTUAL = "factual"
    ANALOGY = "analogy"
    IDEA_GENERATION = "idea_generation"
    COMPARISON = "comparison"
    EXPLORATION = "exploration"


class KnowledgeGraphQuery(BaseModel):
    """Input parameters for querying the knowledge graph."""
    query: str = Field(..., description="The user's query text")
    query_type: QueryType = Field(
        default=QueryType.FACTUAL,
        description="Type of query to perform"
    )
    concept_limit: int = Field(
        default=10,
        description="Maximum number of concepts to retrieve"
    )
    relationship_limit: int = Field(
        default=50,
        description="Maximum number of relationships to retrieve"
    )
    creativity: float = Field(
        default=0.7,
        description="Creativity level (0.0-1.0) for idea generation"
    )
    domains: Optional[List[str]] = Field(
        default=None,
        description="Optional list of domains to focus on"
    )


class AnalogyConcept(BaseModel):
    """A concept pairing for analogies."""
    source_concept: str = Field(..., description="Source concept name")
    target_concept: str = Field(..., description="Target concept name")
    explanation: str = Field(..., description="Explanation of the analogy")
    common_patterns: List[str] = Field(..., description="Common patterns or principles")
    similarity_score: float = Field(..., description="Estimated similarity score (0-1)")


class CrossDomainIdea(BaseModel):
    """A novel idea generated across domains."""
    title: str = Field(..., description="Title of the new idea")
    description: str = Field(..., description="Description of the idea")
    source_concepts: List[str] = Field(..., description="Source concepts that inspired this idea")
    domains_connected: List[str] = Field(..., description="Domains this idea connects")
    potential_applications: List[str] = Field(..., description="Potential applications or use cases")
    novelty_score: float = Field(..., description="Estimated novelty score (0-1)")


class KnowledgeGraphQA:
    """
    A system for querying a knowledge graph using LangChain for:
    - Factual question answering
    - Analogy generation
    - Cross-domain idea creation
    """

    def __init__(
            self,
            neo4j_uri: str,
            neo4j_user: str,
            neo4j_password: str,
            openai_api_key: Optional[str] = None,
            model_name: str = "gpt-4o-mini"
    ):
        """
        Initialize the KnowledgeGraphQA system.

        Args:
            neo4j_uri: URI for the Neo4j database
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            openai_api_key: OpenAI API key (or use environment variable)
            model_name: Name of the LLM model to use
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # Connect to Neo4j
        try:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

        # Initialize LLM
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set as environment variable")

        # Updated initialization to handle compatibility issues
        try:
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0.7
            )
        except TypeError:
            # Fall back to simpler initialization if there are keyword argument issues
            import openai
            openai.api_key = api_key
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.7
            )

        # Initialize chains
        self._init_chains()

    def _init_chains(self):
        """Initialize the various LangChain chains."""
        # Query classifier chain
        self.query_classifier_chain = self._create_query_classifier_chain()

        # Factual QA chain
        self.factual_qa_chain = self._create_factual_qa_chain()

        # Analogy generation chain
        self.analogy_chain = self._create_analogy_chain()

        # Idea generation chain
        self.idea_generation_chain = self._create_idea_generation_chain()

        # Comparison chain
        self.comparison_chain = self._create_comparison_chain()

        # Exploration chain
        self.exploration_chain = self._create_exploration_chain()

    def _create_query_classifier_chain(self):
        """Create a chain to classify the type of query."""
        template = """
        You are an expert at classifying knowledge graph queries.
        Analyze the following query and determine its primary purpose.

        Query: {query}

        Classify this query into EXACTLY ONE of these categories:
        - factual: Seeking specific factual information about concepts
        - analogy: Looking for analogies or similarities between concepts
        - idea_generation: Seeking new ideas or innovations
        - comparison: Requesting a comparison between multiple concepts
        - exploration: Exploring a domain or concept space

        Return ONLY the category name in lowercase, nothing else.
        """

        prompt = ChatPromptTemplate.from_template(template)

        return prompt | self.llm | StrOutputParser()

    def _extract_concepts_from_query(self, query: str) -> List[str]:
        """Extract potential concept names from a query string."""
        template = """
        Extract the main concepts or entities from this query.
        Return them as a JSON list of strings, including only the nouns or entities.

        Query: {query}

        Format your response as a JSON array of strings ONLY.
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | JsonOutputParser()

        try:
            concepts = chain.invoke({"query": query})
            return concepts
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            # Try a fallback with different parser
            try:
                result = prompt | self.llm | StrOutputParser()
                text_result = result.invoke({"query": query})
                # Try to extract JSON array from text
                import re
                match = re.search(r'\[.*\]', text_result, re.DOTALL)
                if match:
                    concepts = json.loads(match.group(0))
                    return concepts
                return []
            except:
                return []

    def _get_concept_by_name(self, name: str) -> Dict[str, Any]:
        """Retrieve a concept from the knowledge graph by name."""
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

    def _search_concepts(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for concepts using semantic search."""
        try:
            # First try direct name match
            exact_concepts = []
            query_concepts = self._extract_concepts_from_query(query_text)

            for concept_name in query_concepts:
                concept = self._get_concept_by_name(concept_name)
                if concept:
                    exact_concepts.append({
                        "name": concept["name"],
                        "description": concept["what"],
                        "score": 1.0  # Perfect match
                    })

            # If we have exact matches, return them
            if exact_concepts:
                return exact_concepts[:limit]

            # Otherwise, fallback to semantic search via Cypher
            # Note: This assumes you have embeddings in your graph
            with self.driver.session() as session:
                # Try to find concepts by keyword matching as fallback
                result = session.run("""
                MATCH (c:Concept)
                WHERE toLower(c.name) CONTAINS toLower($query) 
                   OR toLower(c.what) CONTAINS toLower($query)
                RETURN c.name AS name, c.what AS description, 0.8 AS score
                LIMIT $limit
                """, {"query": query_text, "limit": limit})

                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Error in concept search: {e}")
            return []

    def _get_concept_relationships(self, concept_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get relationships for a concept."""
        try:
            with self.driver.session() as session:
                result = session.run("""
                MATCH (c:Concept {name: $name})-[r]->(related)
                RETURN type(r) AS relation, related.name AS target, r.justification AS justification, 'outgoing' AS direction
                UNION
                MATCH (related)-[r]->(c:Concept {name: $name})
                RETURN type(r) AS relation, related.name AS source, r.justification AS justification, 'incoming' AS direction
                LIMIT $limit
                """, {"name": concept_name, "limit": limit})

                relationships = []
                for record in result:
                    rel_dict = dict(record)
                    relationships.append(rel_dict)

                return relationships
        except Exception as e:
            logger.error(f"Error getting relationships: {e}")
            return []

    def _get_domain_concepts(self, domain: str, limit: int = 20) -> List[str]:
        """Get concepts belonging to a specific domain."""
        try:
            with self.driver.session() as session:
                result = session.run("""
                MATCH (c:Concept)
                WHERE c.domain = $domain
                RETURN c.name AS name
                LIMIT $limit
                """, {"domain": domain, "limit": limit})

                return [record["name"] for record in result]
        except Exception as e:
            logger.error(f"Error getting domain concepts: {e}")
            return []

    def _get_shortest_paths(self, source: str, target: str, max_length: int = 3) -> List[List[Dict[str, Any]]]:
        """Find shortest paths between two concepts."""
        try:
            with self.driver.session() as session:
                result = session.run("""
                MATCH path = shortestPath((a:Concept {name: $source})-[*1..{max_length}]-(b:Concept {name: $target}))
                RETURN path
                LIMIT 3
                """, {"source": source, "target": target, "max_length": max_length})

                paths = []
                for record in result:
                    path = record["path"]
                    path_data = []

                    # Extract nodes and relationships from path
                    nodes = path.nodes
                    relationships = path.relationships

                    for i, rel in enumerate(relationships):
                        step = {
                            "source": nodes[i]["name"],
                            "relation": rel.type,
                            "target": nodes[i + 1]["name"],
                            "justification": rel.get("justification", "")
                        }
                        path_data.append(step)

                    paths.append(path_data)

                return paths
        except Exception as e:
            logger.error(f"Error finding paths: {e}")
            return []

    def _retrieve_knowledge_context(self, query: KnowledgeGraphQuery) -> Dict[str, Any]:
        """Retrieve relevant knowledge context for a query."""
        # Extract concepts from query
        query_concepts = self._extract_concepts_from_query(query.query)

        # Search for relevant concepts
        search_results = []
        for concept_name in query_concepts:
            # Try exact match first
            concept = self._get_concept_by_name(concept_name)
            if concept:
                search_results.append({
                    "name": concept["name"],
                    "description": concept["what"],
                    "score": 1.0
                })
                continue

            # Fall back to search
            results = self._search_concepts(concept_name, limit=3)
            search_results.extend(results)

        # If no direct hits, search with full query
        if not search_results:
            search_results = self._search_concepts(query.query, limit=query.concept_limit)

        # Get detailed info and relationships for each concept
        concepts_data = []
        for result in search_results[:query.concept_limit]:
            concept_name = result["name"]
            concept = self._get_concept_by_name(concept_name)

            if concept:
                # Get relationships
                relationships = self._get_concept_relationships(concept_name, limit=query.relationship_limit)
                concept["relationships"] = relationships
                concepts_data.append(concept)

        # For domain-specific queries, add domain concepts
        domain_concepts = []
        if query.domains:
            for domain in query.domains:
                domain_concepts.extend(self._get_domain_concepts(domain, limit=10))

            # Get data for domain concepts not already included
            existing_names = [c["name"] for c in concepts_data]
            for name in domain_concepts:
                if name not in existing_names:
                    concept = self._get_concept_by_name(name)
                    if concept:
                        relationships = self._get_concept_relationships(name, limit=10)
                        concept["relationships"] = relationships
                        concepts_data.append(concept)

        # For analogy queries, find paths between concepts if we have at least 2
        paths = []
        if query.query_type == QueryType.ANALOGY and len(query_concepts) >= 2:
            paths = self._get_shortest_paths(query_concepts[0], query_concepts[1])

        return {
            "concepts": concepts_data,
            "paths": paths,
            "query_concepts": query_concepts,
            "domain_concepts": domain_concepts
        }

    def _create_factual_qa_chain(self):
        """Create a chain for factual question answering."""
        template = """
        You are an expert knowledge assistant answering questions based on a knowledge graph.

        USER QUERY: {query}

        KNOWLEDGE GRAPH CONTEXT:
        {context}

        Based on the information from the knowledge graph, please answer the user's question.
        If the knowledge graph doesn't contain enough information to answer, say so clearly.
        Cite the specific concepts from the knowledge graph that inform your answer.
        """

        prompt = ChatPromptTemplate.from_template(template)

        return prompt | self.llm | StrOutputParser()

    def _create_analogy_chain(self):
        """Create a chain for generating analogies."""
        template = """
        You are an expert at finding analogies between concepts in different domains.

        USER QUERY: {query}

        KNOWLEDGE GRAPH CONTEXT:
        {context}

        Based on the concepts in the knowledge graph, generate insightful analogies.
        Focus on finding non-obvious but meaningful similarities between concepts,
        especially across different domains.

        Respond with a JSON object containing an array of analogies:
        {{
          "analogies": [
            {{
              "source_concept": "concept1",
              "target_concept": "concept2",
              "explanation": "detailed explanation of the analogy",
              "common_patterns": ["pattern1", "pattern2"],
              "similarity_score": 0.85
            }}
          ]
        }}

        Use ONLY concepts that are present in the knowledge graph context provided.
        """

        prompt = ChatPromptTemplate.from_template(template)

        return prompt | self.llm | JsonOutputParser()

    def _create_idea_generation_chain(self):
        """Create a chain for generating novel ideas."""
        template = """
        You are an expert at generating novel ideas by connecting concepts across domains.

        USER QUERY: {query}

        KNOWLEDGE GRAPH CONTEXT:
        {context}

        Creativity level: {creativity}

        Based on the concepts in the knowledge graph, generate {num_ideas} novel ideas that:
        1. Connect concepts across different domains
        2. Apply principles from one domain to another
        3. Create valuable new combinations or applications

        Respond with a JSON object containing an array of ideas:
        {{
          "ideas": [
            {{
              "title": "Idea title",
              "description": "Detailed description of the idea",
              "source_concepts": ["concept1", "concept2"],
              "domains_connected": ["domain1", "domain2"],
              "potential_applications": ["application1", "application2"],
              "novelty_score": 0.85
            }}
          ]
        }}

        Be bold and innovative in your idea generation while still grounding the ideas
        in the concepts from the knowledge graph.
        """

        prompt = ChatPromptTemplate.from_template(template)

        return prompt | self.llm | JsonOutputParser()

    def _create_comparison_chain(self):
        """Create a chain for comparing concepts."""
        template = """
        You are an expert at comparing and contrasting concepts from a knowledge graph.

        USER QUERY: {query}

        KNOWLEDGE GRAPH CONTEXT:
        {context}

        Based on the concepts in the knowledge graph, provide a detailed comparison
        of the relevant concepts. Include:

        1. Key similarities
        2. Important differences
        3. Complementary aspects
        4. Potential synergies
        5. Historical or evolutionary relationships

        Structure your response as a well-organized comparison that helps the user
        understand the relationships between these concepts.
        """

        prompt = ChatPromptTemplate.from_template(template)

        return prompt | self.llm | StrOutputParser()

    def _create_exploration_chain(self):
        """Create a chain for exploring concept spaces."""
        template = """
        You are an expert at exploring concept spaces and knowledge domains.

        USER QUERY: {query}

        KNOWLEDGE GRAPH CONTEXT:
        {context}

        Based on the concepts in the knowledge graph, provide an exploration of this
        conceptual space. Include:

        1. Core concepts and their relationships
        2. Key principles or patterns
        3. Interesting connections or insights
        4. Potential areas for further exploration
        5. Questions that emerge from analyzing this space

        Structure your response as an engaging exploration that helps the user
        understand this knowledge domain more deeply.
        """

        prompt = ChatPromptTemplate.from_template(template)

        return prompt | self.llm | StrOutputParser()

    def _format_context(self, knowledge_context: Dict[str, Any]) -> str:
        """Format knowledge context for inclusion in prompts."""
        formatted = "CONCEPTS:\n"

        for concept in knowledge_context["concepts"]:
            formatted += f"- {concept['name']}:\n"
            formatted += f"  What: {concept['what']}\n"
            if concept.get("how"):
                formatted += f"  How: {concept['how']}\n"
            if concept.get("why"):
                formatted += f"  Why: {concept['why']}\n"
            if concept.get("domain"):
                formatted += f"  Domain: {concept['domain']}\n"

            # Add relationships
            if concept.get("relationships"):
                formatted += "  Relationships:\n"
                for rel in concept["relationships"]:
                    if "target" in rel:
                        formatted += f"    - {concept['name']} → [{rel['relation']}] → {rel['target']}\n"
                    elif "source" in rel:
                        formatted += f"    - {rel['source']} → [{rel['relation']}] → {concept['name']}\n"

            formatted += "\n"

        # Add paths if available
        if knowledge_context.get("paths"):
            formatted += "PATHS BETWEEN CONCEPTS:\n"
            for i, path in enumerate(knowledge_context["paths"]):
                formatted += f"Path {i + 1}:\n"
                for step in path:
                    formatted += f"  {step['source']} → [{step['relation']}] → {step['target']}\n"
                formatted += "\n"

        return formatted

    def process_query(self, query: Union[str, KnowledgeGraphQuery]) -> Dict[str, Any]:
        """
        Process a user query and return the appropriate response.

        Args:
            query: Either a query string or a KnowledgeGraphQuery object

        Returns:
            Response dictionary with results and metadata
        """
        # Convert string to KnowledgeGraphQuery if needed
        if isinstance(query, str):
            # Classify the query type
            try:
                query_type_str = self.query_classifier_chain.invoke({"query": query})
                query_type = QueryType(query_type_str)
            except:
                # Default to factual if classification fails
                query_type = QueryType.FACTUAL

            # Create query object
            query_obj = KnowledgeGraphQuery(
                query=query,
                query_type=query_type
            )
        else:
            query_obj = query

        # Retrieve knowledge context
        knowledge_context = self._retrieve_knowledge_context(query_obj)
        formatted_context = self._format_context(knowledge_context)

        # Choose the appropriate chain based on query type
        if query_obj.query_type == QueryType.FACTUAL:
            result = self.factual_qa_chain.invoke({
                "query": query_obj.query,
                "context": formatted_context
            })
            response = {
                "answer": result,
                "type": "factual",
                "sources": [c["name"] for c in knowledge_context["concepts"]]
            }

        elif query_obj.query_type == QueryType.ANALOGY:
            result = self.analogy_chain.invoke({
                "query": query_obj.query,
                "context": formatted_context
            })
            response = {
                "analogies": result.get("analogies", []),
                "type": "analogy"
            }

        elif query_obj.query_type == QueryType.IDEA_GENERATION:
            result = self.idea_generation_chain.invoke({
                "query": query_obj.query,
                "context": formatted_context,
                "creativity": query_obj.creativity,
                "num_ideas": 3  # Default to 3 ideas
            })
            response = {
                "ideas": result.get("ideas", []),
                "type": "idea_generation"
            }

        elif query_obj.query_type == QueryType.COMPARISON:
            result = self.comparison_chain.invoke({
                "query": query_obj.query,
                "context": formatted_context
            })
            response = {
                "comparison": result,
                "type": "comparison",
                "concepts_compared": knowledge_context["query_concepts"]
            }

        elif query_obj.query_type == QueryType.EXPLORATION:
            result = self.exploration_chain.invoke({
                "query": query_obj.query,
                "context": formatted_context
            })
            response = {
                "exploration": result,
                "type": "exploration",
                "concepts_explored": [c["name"] for c in knowledge_context["concepts"]]
            }

        # Add metadata to the response
        response["metadata"] = {
            "concepts_retrieved": len(knowledge_context["concepts"]),
            "query_type": query_obj.query_type,
            "query": query_obj.query
        }

        return response

    def find_analogies(self, source_concept: str, target_domain: Optional[str] = None, limit: int = 3) -> List[
        AnalogyConcept]:
        """
        Find analogies between a source concept and concepts in a target domain.

        Args:
            source_concept: The source concept to find analogies for
            target_domain: Optional domain to look for analogies in
            limit: Maximum number of analogies to return

        Returns:
            List of AnalogyConcept objects
        """
        # Get source concept data
        source_data = self._get_concept_by_name(source_concept)
        if not source_data:
            logger.warning(f"Source concept '{source_concept}' not found")
            return []

        # Get target domain concepts if specified
        target_concepts = []
        if target_domain:
            target_concepts = self._get_domain_concepts(target_domain, limit=20)

        # If no target domain or no concepts found, get other concepts
        if not target_concepts:
            # Get concepts from different domains
            with self.driver.session() as session:
                result = session.run("""
                MATCH (c:Concept)
                WHERE c.domain <> $source_domain AND c.domain <> ''
                RETURN c.name AS name, c.domain AS domain
                LIMIT 20
                """, {"source_domain": source_data.get("domain", "")})

                target_concepts = [record["name"] for record in result]

        # If still no target concepts, we can't find analogies
        if not target_concepts:
            logger.warning("No target concepts found for analogy generation")
            return []

        # Get relationships for source concept
        source_relationships = self._get_concept_relationships(source_concept, limit=30)

        # Format context for the LLM
        context = f"SOURCE CONCEPT:\n"
        context += f"Name: {source_data['name']}\n"
        context += f"Definition: {source_data['what']}\n"
        context += f"Domain: {source_data['domain']}\n"

        if source_relationships:
            context += "Relationships:\n"
            for rel in source_relationships:
                if "target" in rel:
                    context += f"- {source_data['name']} → [{rel['relation']}] → {rel['target']}\n"
                elif "source" in rel:
                    context += f"- {rel['source']} → [{rel['relation']}] → {source_data['name']}\n"

        context += "\nTARGET CONCEPTS:\n"
        for target_name in target_concepts:
            target_data = self._get_concept_by_name(target_name)
            if target_data:
                context += f"- {target_data['name']} ({target_data['domain']}): {target_data['what']}\n"

        # Create analogy prompt
        template = """
        You are an expert at finding meaningful analogies between concepts across different domains.

        Find {limit} insightful analogies between the source concept and the target concepts.
        Focus on deep structural similarities and non-obvious connections that reveal shared principles.

        {context}

        For each analogy:
        1. Select the most fitting target concept
        2. Explain the non-obvious connection
        3. Identify the common patterns or principles
        4. Estimate a similarity score (0-1)

        Respond with a JSON object:
        {{
          "analogies": [
            {{
              "source_concept": "{source_concept}",
              "target_concept": "target_name",
              "explanation": "detailed explanation",
              "common_patterns": ["pattern1", "pattern2", "pattern3"],
              "similarity_score": 0.85
            }}
          ]
        }}

        Choose target concepts that create the most insightful and surprising analogies.
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | JsonOutputParser()

        try:
            result = chain.invoke({
                "source_concept": source_concept,
                "context": context,
                "limit": limit
            })

            analogies = result.get("analogies", [])
            return [AnalogyConcept(**analogy) for analogy in analogies]
        except Exception as e:
            logger.error(f"Error generating analogies: {e}")
            return []

    def generate_cross_domain_ideas(
            self,
            domains: List[str] = None,
            concepts: List[str] = None,
            creativity: float = 0.7,
            num_ideas: int = 3
    ) -> List[CrossDomainIdea]:
        """
        Generate novel ideas that connect concepts across domains.

        Args:
            domains: Optional list of domains to focus on
            concepts: Optional list of specific concepts to use
            creativity: Creativity level (0.0-1.0)
            num_ideas: Number of ideas to generate

        Returns:
            List of CrossDomainIdea objects
        """
        # Get domain concepts if domains specified
        domain_concepts = {}
        if domains:
            for domain in domains:
                concepts_list = self._get_domain_concepts(domain, limit=10)
                if concepts_list:
                    domain_concepts[domain] = concepts_list

        # If no domains or domains have no concepts, get random domains
        if not domain_concepts:
            try:
                with self.driver.session() as session:
                    result = session.run("""
                    MATCH (c:Concept)
                    WHERE c.domain IS NOT NULL AND c.domain <> ''
                    WITH DISTINCT c.domain AS domain
                    LIMIT 5
                    MATCH (c:Concept)
                    WHERE c.domain = domain
                    RETURN c.domain AS domain, collect(c.name)[..10] AS concepts
                    """)

                    for record in result:
                        domain = record["domain"]
                        domain_concepts[domain] = record["concepts"]
            except Exception as e:
                logger.error(f"Error getting random domains: {e}")

        # Add specific concepts if provided
        specific_concepts_data = []
        if concepts:
            for concept_name in concepts:
                concept_data = self._get_concept_by_name(concept_name)
                if concept_data:
                    specific_concepts_data.append(concept_data)

        # Prepare context for idea generation
        context = "DOMAINS AND CONCEPTS:\n"
        for domain, concept_list in domain_concepts.items():
            context += f"\nDomain: {domain}\n"
            for concept_name in concept_list:
                concept_data = self._get_concept_by_name(concept_name)
                if concept_data:
                    context += f"- {concept_name}: {concept_data['what']}\n"

        if specific_concepts_data:
            context += "\nSPECIFIC CONCEPTS:\n"
            for concept in specific_concepts_data:
                context += f"- {concept['name']} ({concept['domain']}): {concept['what']}\n"

                # Add some relationships for context
                relationships = self._get_concept_relationships(concept['name'], limit=5)
                if relationships:
                    context += "  Relationships:\n"
                    for rel in relationships:
                        if "target" in rel:
                            context += f"  - {concept['name']} → [{rel['relation']}] → {rel['target']}\n"
                        elif "source" in rel:
                            context += f"  - {rel['source']} → [{rel['relation']}] → {concept['name']}\n"

        # Create idea generation prompt
        template = """
        You are a visionary innovator who creates breakthrough ideas by connecting concepts across different domains.

        Your task is to generate {num_ideas} novel and valuable ideas that connect concepts across different domains.
        These ideas should be innovative, practical, and leverage the unique insights from combining knowledge domains.

        KNOWLEDGE CONTEXT:
        {context}

        CREATIVITY LEVEL: {creativity} (0.0 = conservative, 1.0 = highly speculative)

        For each idea:
        1. Create a compelling title
        2. Provide a clear description
        3. Identify the source concepts that inspired it
        4. Specify which domains are being connected
        5. List potential applications or use cases
        6. Assign a novelty score (0-1)

        Respond with a JSON object:
        {{
          "ideas": [
            {{
              "title": "Idea title",
              "description": "Detailed description of the idea",
              "source_concepts": ["concept1", "concept2"],
              "domains_connected": ["domain1", "domain2"],
              "potential_applications": ["application1", "application2"],
              "novelty_score": 0.85
            }}
          ]
        }}

        With creativity level {creativity}, be {creativity_guidance}.
        Focus on ideas that are both innovative and potentially valuable.
        """

        # Adjust creativity guidance based on level
        if creativity < 0.3:
            creativity_guidance = "pragmatic and focus on immediately applicable ideas"
        elif creativity < 0.7:
            creativity_guidance = "balanced between practical applications and novel approaches"
        else:
            creativity_guidance = "bold and speculative, pushing boundaries with breakthrough thinking"

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm.with_options(temperature=creativity) | JsonOutputParser()

        try:
            result = chain.invoke({
                "context": context,
                "num_ideas": num_ideas,
                "creativity": creativity,
                "creativity_guidance": creativity_guidance
            })

            ideas = result.get("ideas", [])
            return [CrossDomainIdea(**idea) for idea in ideas]
        except Exception as e:
            logger.error(f"Error generating ideas: {e}")
            return []

    def close(self):
        """Close the Neo4j connection."""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()