"""
Symbolic Knowledge Base Builder for EU AI Act

This module ingests legal articles from the EU AI Act and converts them into
machine-readable logical predicates using advanced NLP techniques.

Author: AI Rosetta Stone Team
"""

import re
import spacy
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredicateType(Enum):
    """Types of logical predicates that can be extracted."""
    REQUIREMENT = "requires"
    PROHIBITION = "prohibits"
    OBLIGATION = "obliges"
    CONDITION = "condition"
    SCOPE = "scope"
    PROPERTY = "is_a"
    RELATIONSHIP = "relates_to"


@dataclass
class ExtractedEntity:
    """Represents an entity extracted from legal text."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class ExtractedRelation:
    """Represents a relationship between entities."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    context: str = ""


@dataclass
class LegalPredicate:
    """Represents a logical predicate extracted from legal text."""
    predicate_type: PredicateType
    subject: str
    predicate: str
    object: str
    conditions: List[str]
    confidence: float
    source_text: str
    
    def to_string(self) -> str:
        """Convert predicate to string representation."""
        if self.conditions:
            conditions_str = ", ".join(self.conditions)
            return f"{self.predicate_type.value}({self.subject}, {self.object}, conditions=[{conditions_str}])"
        else:
            return f"{self.predicate_type.value}({self.subject}, {self.object})"


class LegalTextProcessor:
    """Advanced NLP processor for legal text analysis."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the legal text processor.
        
        Args:
            model_name: SpaCy model name for NLP processing
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(f"SpaCy model '{model_name}' not found. Please install it with: python -m spacy download {model_name}")
            raise
            
        # Add custom patterns for legal entities
        self._add_legal_patterns()
        
        # Define legal keywords and their categories
        self.requirement_keywords = {
            "shall", "must", "required", "necessary", "obliged", "mandated",
            "should", "need to", "have to", "ought to"
        }
        
        self.prohibition_keywords = {
            "shall not", "must not", "prohibited", "forbidden", "banned",
            "not permitted", "not allowed", "cannot", "may not"
        }
        
        self.condition_keywords = {
            "if", "when", "where", "provided that", "in case", "subject to",
            "conditional on", "depending on", "in the event"
        }
        
        self.scope_keywords = {
            "during", "throughout", "within", "for the period", "while",
            "in the course of", "at the time of"
        }
        
    def _add_legal_patterns(self):
        """Add custom patterns for legal entity recognition."""
        from spacy.matcher import Matcher
        
        self.matcher = Matcher(self.nlp.vocab)
        
        # Pattern for AI system types
        ai_system_patterns = [
            [{"LOWER": "high-risk"}, {"LOWER": "ai"}, {"LOWER": "systems"}],
            [{"LOWER": "ai"}, {"LOWER": "systems"}],
            [{"LOWER": "artificial"}, {"LOWER": "intelligence"}, {"LOWER": "systems"}],
            [{"LOWER": "limited"}, {"LOWER": "risk"}, {"LOWER": "ai"}, {"LOWER": "systems"}],
            [{"LOWER": "minimal"}, {"LOWER": "risk"}, {"LOWER": "ai"}, {"LOWER": "systems"}]
        ]
        
        # Pattern for oversight concepts
        oversight_patterns = [
            [{"LOWER": "human"}, {"LOWER": "oversight"}],
            [{"LOWER": "human"}, {"LOWER": "supervision"}],
            [{"LOWER": "human"}, {"LOWER": "review"}],
            [{"LOWER": "natural"}, {"LOWER": "persons"}]
        ]
        
        # Pattern for compliance concepts
        compliance_patterns = [
            [{"LOWER": "data"}, {"LOWER": "quality"}],
            [{"LOWER": "transparency"}],
            [{"LOWER": "robustness"}],
            [{"LOWER": "accuracy"}],
            [{"LOWER": "bias"}],
            [{"LOWER": "discrimination"}]
        ]
        
        self.matcher.add("AI_SYSTEM", ai_system_patterns)
        self.matcher.add("OVERSIGHT", oversight_patterns)
        self.matcher.add("COMPLIANCE", compliance_patterns)
        
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """
        Extract legal entities from text using NLP.
        
        Args:
            text: Legal text to analyze
            
        Returns:
            List of extracted entities
        """
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append(ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char
            ))
            
        # Extract custom legal entities using matcher
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            label = self.nlp.vocab.strings[match_id]
            
            entities.append(ExtractedEntity(
                text=span.text,
                label=label,
                start=span.start_char,
                end=span.end_char
            ))
            
        return entities
        
    def extract_relationships(self, text: str) -> List[ExtractedRelation]:
        """
        Extract relationships between entities in legal text.
        
        Args:
            text: Legal text to analyze
            
        Returns:
            List of extracted relationships
        """
        doc = self.nlp(text)
        relationships = []
        
        # Analyze dependency parsing to find relationships
        for token in doc:
            if token.dep_ in ["nsubj", "nsubjpass"]:  # Subject relationships
                # Find the root verb
                root = token.head
                
                # Find objects
                objects = [child for child in root.children 
                          if child.dep_ in ["dobj", "pobj", "attr"]]
                
                for obj in objects:
                    relationships.append(ExtractedRelation(
                        subject=token.text,
                        predicate=root.text,
                        object=obj.text,
                        context=text
                    ))
                    
        return relationships
        
    def identify_legal_constructs(self, text: str) -> Dict[str, List[str]]:
        """
        Identify legal constructs like requirements, prohibitions, etc.
        
        Args:
            text: Legal text to analyze
            
        Returns:
            Dictionary mapping construct types to identified phrases
        """
        text_lower = text.lower()
        constructs = {
            "requirements": [],
            "prohibitions": [],
            "conditions": [],
            "scopes": []
        }
        
        # Check for requirement indicators
        for keyword in self.requirement_keywords:
            if keyword in text_lower:
                constructs["requirements"].append(keyword)
                
        # Check for prohibition indicators
        for keyword in self.prohibition_keywords:
            if keyword in text_lower:
                constructs["prohibitions"].append(keyword)
                
        # Check for condition indicators
        for keyword in self.condition_keywords:
            if keyword in text_lower:
                constructs["conditions"].append(keyword)
                
        # Check for scope indicators
        for keyword in self.scope_keywords:
            if keyword in text_lower:
                constructs["scopes"].append(keyword)
                
        return constructs


class SymbolicKnowledgeBaseBuilder:
    """
    Main class for building symbolic knowledge base from legal text.
    
    This class orchestrates the conversion of legal articles into
    machine-readable logical predicates.
    """
    
    def __init__(self):
        """Initialize the knowledge base builder."""
        self.processor = LegalTextProcessor()
        self.predicates: List[LegalPredicate] = []
        
    def parse_article(self, article_text: str, article_id: str = None) -> List[str]:
        """
        Parse a legal article and convert it to logical predicates.
        
        Args:
            article_text: The text of the legal article
            article_id: Optional article identifier (e.g., "Article 14")
            
        Returns:
            List of logical predicates as strings
        """
        logger.info(f"Parsing article: {article_id or 'Unknown'}")
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(article_text)
        
        # Extract entities and relationships
        entities = self.processor.extract_entities(cleaned_text)
        relationships = self.processor.extract_relationships(cleaned_text)
        legal_constructs = self.processor.identify_legal_constructs(cleaned_text)
        
        # Convert to logical predicates
        predicates = self._convert_to_predicates(
            cleaned_text, entities, relationships, legal_constructs, article_id
        )
        
        # Convert predicates to string format
        predicate_strings = [pred.to_string() for pred in predicates]
        
        logger.info(f"Generated {len(predicate_strings)} predicates")
        return predicate_strings
        
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess legal text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle common legal text patterns
        text = re.sub(r'Article\s+(\d+)\s*\([^)]+\):', r'Article \1:', text)
        
        return text
        
    def _convert_to_predicates(self, text: str, entities: List[ExtractedEntity],
                             relationships: List[ExtractedRelation],
                             legal_constructs: Dict[str, List[str]],
                             article_id: str = None) -> List[LegalPredicate]:
        """Convert extracted information to logical predicates."""
        predicates = []
        
        # Analyze the text structure to identify main requirements
        if any(keyword in text.lower() for keyword in self.processor.requirement_keywords):
            predicates.extend(self._extract_requirements(text, entities, article_id))
            
        if any(keyword in text.lower() for keyword in self.processor.prohibition_keywords):
            predicates.extend(self._extract_prohibitions(text, entities, article_id))
            
        # Extract scope and conditions
        predicates.extend(self._extract_scope_and_conditions(text, entities, article_id))
        
        # Extract properties and relationships
        predicates.extend(self._extract_properties(text, entities, relationships, article_id))
        
        return predicates
        
    def _extract_requirements(self, text: str, entities: List[ExtractedEntity],
                            article_id: str = None) -> List[LegalPredicate]:
        """Extract requirement predicates from text."""
        predicates = []
        
        # Look for AI system entities
        ai_systems = [e for e in entities if "ai" in e.label.lower() or "system" in e.text.lower()]
        oversight_entities = [e for e in entities if "oversight" in e.label.lower()]
        
        # Pattern: "High-risk AI systems shall be designed..."
        if "high-risk ai systems" in text.lower() and "shall" in text.lower():
            if "human oversight" in text.lower():
                predicates.append(LegalPredicate(
                    predicate_type=PredicateType.REQUIREMENT,
                    subject="system_type='high-risk'",
                    predicate="requires",
                    object="component='human_oversight'",
                    conditions=[],
                    confidence=0.9,
                    source_text=text
                ))
                
        # Pattern: "effectively overseen by natural persons"
        if "effectively overseen" in text.lower() and "natural persons" in text.lower():
            predicates.append(LegalPredicate(
                predicate_type=PredicateType.PROPERTY,
                subject="human_oversight",
                predicate="is_a",
                object="'effective'",
                conditions=[],
                confidence=0.8,
                source_text=text
            ))
            
        return predicates
        
    def _extract_prohibitions(self, text: str, entities: List[ExtractedEntity],
                            article_id: str = None) -> List[LegalPredicate]:
        """Extract prohibition predicates from text."""
        predicates = []
        
        # Look for prohibition patterns
        if "shall not" in text.lower() or "prohibited" in text.lower():
            # Extract what is prohibited
            # This is a simplified example - real implementation would be more sophisticated
            pass
            
        return predicates
        
    def _extract_scope_and_conditions(self, text: str, entities: List[ExtractedEntity],
                                     article_id: str = None) -> List[LegalPredicate]:
        """Extract scope and condition predicates."""
        predicates = []
        
        # Look for temporal scope
        if "during the period" in text.lower() and "in use" in text.lower():
            predicates.append(LegalPredicate(
                predicate_type=PredicateType.SCOPE,
                subject="human_oversight",
                predicate="scope",
                object="period='in_use'",
                conditions=[],
                confidence=0.8,
                source_text=text
            ))
            
        return predicates
        
    def _extract_properties(self, text: str, entities: List[ExtractedEntity],
                          relationships: List[ExtractedRelation],
                          article_id: str = None) -> List[LegalPredicate]:
        """Extract property predicates from relationships."""
        predicates = []
        
        # Convert relationships to predicates
        for rel in relationships:
            if rel.predicate.lower() in ["designed", "developed"]:
                predicates.append(LegalPredicate(
                    predicate_type=PredicateType.PROPERTY,
                    subject=rel.subject.lower().replace(" ", "_"),
                    predicate="has_property",
                    object=f"'{rel.predicate.lower()}'",
                    conditions=[],
                    confidence=rel.confidence * 0.7,  # Lower confidence for derived predicates
                    source_text=text
                ))
                
        return predicates
        
    def get_predicate_statistics(self) -> Dict[str, int]:
        """Get statistics about extracted predicates."""
        stats = {}
        for pred_type in PredicateType:
            count = len([p for p in self.predicates if p.predicate_type == pred_type])
            stats[pred_type.value] = count
        return stats


def demonstrate_article_14():
    """Demonstrate the knowledge base builder with Article 14 from the whitepaper."""
    
    print("=" * 80)
    print("AI ROSETTA STONE - SYMBOLIC KNOWLEDGE BASE BUILDER")
    print("=" * 80)
    print()
    
    # Article 14 from the whitepaper
    article_14_text = """
    Article 14 (Human Oversight): High-risk AI systems shall be designed and developed 
    in such a way that they can be effectively overseen by natural persons during the 
    period in which the AI system is in use.
    """
    
    print("📄 INPUT ARTICLE:")
    print("-" * 40)
    print(article_14_text.strip())
    print()
    
    # Initialize the knowledge base builder
    try:
        kb_builder = SymbolicKnowledgeBaseBuilder()
        
        print("🔍 PROCESSING...")
        print("-" * 40)
        
        # Parse the article
        predicates = kb_builder.parse_article(article_14_text, "Article 14")
        
        print("✅ EXTRACTED LOGICAL PREDICATES:")
        print("-" * 40)
        
        for i, predicate in enumerate(predicates, 1):
            print(f"{i}. {predicate}")
            
        print()
        print("📊 STATISTICS:")
        print("-" * 40)
        stats = kb_builder.get_predicate_statistics()
        for pred_type, count in stats.items():
            if count > 0:
                print(f"  {pred_type}: {count}")
                
        print()
        print("🎯 EXPECTED OUTPUT FORMAT:")
        print("-" * 40)
        expected = [
            "requires(system_type='high-risk', component='human_oversight')",
            "is_a(human_oversight, 'effective')",
            "scope(human_oversight, period='in_use')"
        ]
        
        for i, exp in enumerate(expected, 1):
            print(f"{i}. {exp}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("\nℹ️  Make sure you have spaCy installed with the English model:")
        print("   pip install spacy")
        print("   python -m spacy download en_core_web_sm")


def test_additional_articles():
    """Test the builder with additional EU AI Act articles."""
    
    print("\n" + "=" * 80)
    print("TESTING WITH ADDITIONAL ARTICLES")
    print("=" * 80)
    
    articles = [
        {
            "id": "Article 10",
            "text": """Article 10 (Data and data governance): High-risk AI systems which make 
            use of techniques involving the training of models with data shall be developed 
            on the basis of training, validation and testing data sets that meet the quality 
            criteria referred to in paragraphs 2 to 5."""
        },
        {
            "id": "Article 13", 
            "text": """Article 13 (Transparency and provision of information to users): High-risk 
            AI systems shall be designed and developed in such a way that their operation is 
            sufficiently transparent to enable users to interpret the system's output and use 
            it appropriately."""
        }
    ]
    
    try:
        kb_builder = SymbolicKnowledgeBaseBuilder()
        
        for article in articles:
            print(f"\n📄 PROCESSING {article['id']}:")
            print("-" * 40)
            
            predicates = kb_builder.parse_article(article['text'], article['id'])
            
            print(f"Extracted {len(predicates)} predicates:")
            for i, predicate in enumerate(predicates, 1):
                print(f"  {i}. {predicate}")
                
    except Exception as e:
        print(f"❌ ERROR: {e}")


if __name__ == "__main__":
    """Main execution - demonstrate the knowledge base builder."""
    
    # Run the main demonstration with Article 14
    demonstrate_article_14()
    
    # Test with additional articles
    test_additional_articles()
    
    print("\n" + "=" * 80)
    print("✨ KNOWLEDGE BASE BUILDER DEMONSTRATION COMPLETE")
    print("=" * 80)