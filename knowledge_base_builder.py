"""
AI Rosetta Stone - Symbolic Knowledge Base Builder

This module implements the Symbolic Knowledge Base component of the AI Rosetta Stone
engine as described in the technical whitepaper. It ingests legal articles from the
EU AI Act and converts them into machine-readable logical predicates.

Author: AI Rosetta Stone Team
Purpose: De-risking AI Deployment through Neuro-Symbolic Regulatory Compliance
"""

import re
import spacy
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class EntityType(Enum):
    """Enumeration of entity types relevant to EU AI Act articles."""
    SYSTEM_TYPE = "system_type"
    REQUIREMENT = "requirement"
    COMPONENT = "component"
    SCOPE = "scope"
    CONDITION = "condition"
    ATTRIBUTE = "attribute"
    PERSON_TYPE = "person_type"
    TIME_PERIOD = "time_period"


class RelationType(Enum):
    """Enumeration of relationship types in legal text."""
    REQUIRES = "requires"
    PROHIBITS = "prohibits"
    IS_A = "is_a"
    SCOPE = "scope"
    APPLIES_TO = "applies_to"
    DURING = "during"
    DESIGNED_FOR = "designed_for"
    OVERSEEN_BY = "overseen_by"


@dataclass
class Entity:
    """Represents an extracted entity from legal text."""
    text: str
    entity_type: EntityType
    normalized_form: str
    confidence: float = 1.0


@dataclass
class Relationship:
    """Represents an extracted relationship between entities."""
    subject: Entity
    relation_type: RelationType
    object: Entity
    confidence: float = 1.0


class KnowledgeBaseBuilder:
    """
    Main class for building the Symbolic Knowledge Base from legal articles.
    
    This class uses NLP techniques to extract entities and relationships from
    legal text and converts them into logical predicates suitable for automated
    reasoning and compliance checking.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the Knowledge Base Builder.
        
        Args:
            model_name: spaCy model name to use for NLP processing
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Warning: {model_name} not found. Using blank English model.")
            self.nlp = spacy.blank("en")
        
        # Define domain-specific patterns for EU AI Act
        self._setup_patterns()
        
    def _setup_patterns(self):
        """Set up domain-specific patterns for entity and relationship extraction."""
        
        # System type patterns
        self.system_type_patterns = [
            r"high-risk\s+AI\s+system",
            r"high-risk\s+system",
            r"AI\s+system",
            r"artificial\s+intelligence\s+system",
            r"system"
        ]
        
        # Requirement patterns
        self.requirement_patterns = [
            r"human\s+oversight",
            r"transparency",
            r"robustness",
            r"data\s+quality",
            r"accuracy",
            r"fairness",
            r"accountability",
            r"explainability"
        ]
        
        # Modal verbs indicating requirements
        self.modal_requirements = [
            "shall", "must", "should", "ought to", "required to",
            "obliged to", "mandated to", "compelled to"
        ]
        
        # Prohibition patterns
        self.prohibition_patterns = [
            "shall not", "must not", "prohibited", "forbidden",
            "not allowed", "not permitted"
        ]
        
        # Time/scope patterns
        self.scope_patterns = [
            r"during\s+the\s+period",
            r"in\s+use",
            r"throughout\s+the\s+lifecycle",
            r"at\s+all\s+times",
            r"continuously"
        ]
        
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from legal text using NLP and pattern matching.
        
        Args:
            text: Input legal article text
            
        Returns:
            List of extracted entities
        """
        entities = []
        doc = self.nlp(text)
        
        # Extract system types
        for pattern in self.system_type_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group()
                normalized = self._normalize_system_type(entity_text)
                entities.append(Entity(
                    text=entity_text,
                    entity_type=EntityType.SYSTEM_TYPE,
                    normalized_form=normalized,
                    confidence=0.9
                ))
        
        # Extract requirements/components
        for pattern in self.requirement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group()
                normalized = self._normalize_requirement(entity_text)
                entities.append(Entity(
                    text=entity_text,
                    entity_type=EntityType.COMPONENT,
                    normalized_form=normalized,
                    confidence=0.85
                ))
        
        # Extract scope/time patterns
        for pattern in self.scope_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group()
                normalized = self._normalize_scope(entity_text)
                entities.append(Entity(
                    text=entity_text,
                    entity_type=EntityType.TIME_PERIOD,
                    normalized_form=normalized,
                    confidence=0.8
                ))
        
        # Extract person types using spaCy NER
        for ent in doc.ents:
            if ent.label_ == "PERSON" or "person" in ent.text.lower():
                normalized = self._normalize_person_type(ent.text)
                entities.append(Entity(
                    text=ent.text,
                    entity_type=EntityType.PERSON_TYPE,
                    normalized_form=normalized,
                    confidence=0.7
                ))
        
        # Remove duplicates based on normalized form
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.normalized_form not in seen:
                seen.add(entity.normalized_form)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """
        Extract relationships between entities in the legal text.
        
        Args:
            text: Input legal article text
            entities: List of previously extracted entities
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        doc = self.nlp(text)
        
        # Create entity lookup by text
        entity_lookup = {ent.text.lower(): ent for ent in entities}
        
        # Pattern-based relationship extraction
        
        # "shall be designed" pattern - indicates requirement
        design_pattern = r"(\w+(?:\s+\w+)*)\s+shall\s+be\s+designed.*?(?:that\s+they\s+can\s+be|to\s+be|with)\s+(.*?)(?:\.|;|$)"
        matches = re.finditer(design_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            subject_text = match.group(1).strip()
            object_text = match.group(2).strip()
            
            subject_entity = self._find_or_create_entity(subject_text, entities, EntityType.SYSTEM_TYPE)
            object_entity = self._find_or_create_entity(object_text, entities, EntityType.COMPONENT)
            
            if subject_entity and object_entity:
                relationships.append(Relationship(
                    subject=subject_entity,
                    relation_type=RelationType.REQUIRES,
                    object=object_entity,
                    confidence=0.9
                ))
        
        # "overseen by" pattern
        oversight_pattern = r"(overseen|supervised)\s+by\s+(.*?)(?:\s+during|\s+in|\.|;|$)"
        matches = re.finditer(oversight_pattern, text, re.IGNORECASE)
        
        for match in matches:
            overseer_text = match.group(2).strip()
            overseer_entity = self._find_or_create_entity(overseer_text, entities, EntityType.PERSON_TYPE)
            
            # Find system entity to be overseen
            system_entities = [e for e in entities if e.entity_type == EntityType.SYSTEM_TYPE]
            if system_entities and overseer_entity:
                relationships.append(Relationship(
                    subject=system_entities[0],  # Use first system found
                    relation_type=RelationType.OVERSEEN_BY,
                    object=overseer_entity,
                    confidence=0.85
                ))
        
        # "during the period" scope relationships
        scope_pattern = r"during\s+the\s+period.*?(in\s+which.*?)(?:\.|;|$)"
        matches = re.finditer(scope_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            scope_text = match.group(1).strip()
            scope_entity = self._find_or_create_entity(scope_text, entities, EntityType.TIME_PERIOD)
            
            # Find components to apply scope to
            component_entities = [e for e in entities if e.entity_type == EntityType.COMPONENT]
            for comp_entity in component_entities:
                if scope_entity:
                    relationships.append(Relationship(
                        subject=comp_entity,
                        relation_type=RelationType.SCOPE,
                        object=scope_entity,
                        confidence=0.8
                    ))
        
        return relationships
    
    def convert_to_predicates(self, entities: List[Entity], relationships: List[Relationship]) -> List[str]:
        """
        Convert extracted entities and relationships into logical predicates.
        
        Args:
            entities: List of extracted entities
            relationships: List of extracted relationships
            
        Returns:
            List of logical predicates as strings
        """
        predicates = []
        
        # Convert relationships to predicates
        for rel in relationships:
            if rel.relation_type == RelationType.REQUIRES:
                predicate = f"requires({self._entity_to_param(rel.subject)}, {self._entity_to_param(rel.object)})"
                predicates.append(predicate)
                
            elif rel.relation_type == RelationType.OVERSEEN_BY:
                predicate = f"overseen_by({self._entity_to_param(rel.subject)}, {self._entity_to_param(rel.object)})"
                predicates.append(predicate)
                
            elif rel.relation_type == RelationType.SCOPE:
                predicate = f"scope({self._entity_to_param(rel.subject)}, {self._entity_to_param(rel.object)})"
                predicates.append(predicate)
                
            elif rel.relation_type == RelationType.IS_A:
                predicate = f"is_a({self._entity_to_param(rel.subject)}, {self._entity_to_param(rel.object)})"
                predicates.append(predicate)
        
        # Add standalone entity predicates for important attributes
        for entity in entities:
            if entity.entity_type == EntityType.COMPONENT and "effective" in entity.text.lower():
                predicate = f"is_a({entity.normalized_form}, 'effective')"
                predicates.append(predicate)
        
        return predicates
    
    def process_article(self, article_text: str) -> Dict:
        """
        Main method to process a legal article and extract logical predicates.
        
        Args:
            article_text: Input legal article text
            
        Returns:
            Dictionary containing entities, relationships, and predicates
        """
        # Extract entities
        entities = self.extract_entities(article_text)
        
        # Extract relationships
        relationships = self.extract_relationships(article_text, entities)
        
        # Convert to predicates
        predicates = self.convert_to_predicates(entities, relationships)
        
        return {
            "entities": entities,
            "relationships": relationships,
            "predicates": predicates,
            "article_text": article_text
        }
    
    def _normalize_system_type(self, text: str) -> str:
        """Normalize system type entity."""
        text = text.lower().strip()
        if "high-risk" in text:
            return "high-risk"
        elif "ai system" in text or "artificial intelligence system" in text:
            return "ai_system"
        else:
            return "system"
    
    def _normalize_requirement(self, text: str) -> str:
        """Normalize requirement/component entity."""
        text = text.lower().strip()
        return text.replace(" ", "_").replace("-", "_")
    
    def _normalize_scope(self, text: str) -> str:
        """Normalize scope/time period entity."""
        text = text.lower().strip()
        if "in use" in text:
            return "in_use"
        elif "during the period" in text:
            return "usage_period"
        else:
            return text.replace(" ", "_").replace("-", "_")
    
    def _normalize_person_type(self, text: str) -> str:
        """Normalize person type entity."""
        text = text.lower().strip()
        if "natural person" in text:
            return "natural_persons"
        else:
            return text.replace(" ", "_").replace("-", "_")
    
    def _find_or_create_entity(self, text: str, entities: List[Entity], entity_type: EntityType) -> Optional[Entity]:
        """Find existing entity or create new one."""
        text = text.strip()
        
        # Try to find existing entity
        for entity in entities:
            if entity.text.lower() == text.lower() or text.lower() in entity.text.lower():
                return entity
        
        # Create new entity if not found
        if entity_type == EntityType.SYSTEM_TYPE:
            normalized = self._normalize_system_type(text)
        elif entity_type == EntityType.COMPONENT:
            normalized = self._normalize_requirement(text)
        elif entity_type == EntityType.TIME_PERIOD:
            normalized = self._normalize_scope(text)
        elif entity_type == EntityType.PERSON_TYPE:
            normalized = self._normalize_person_type(text)
        else:
            normalized = text.lower().replace(" ", "_")
        
        new_entity = Entity(
            text=text,
            entity_type=entity_type,
            normalized_form=normalized,
            confidence=0.7
        )
        entities.append(new_entity)
        return new_entity
    
    def _entity_to_param(self, entity: Entity) -> str:
        """Convert entity to parameter string for predicates."""
        if entity.entity_type == EntityType.SYSTEM_TYPE:
            return f"system_type='{entity.normalized_form}'"
        elif entity.entity_type == EntityType.COMPONENT:
            return f"component='{entity.normalized_form}'"
        elif entity.entity_type == EntityType.TIME_PERIOD:
            return f"period='{entity.normalized_form}'"
        elif entity.entity_type == EntityType.PERSON_TYPE:
            return f"overseer='{entity.normalized_form}'"
        else:
            return f"'{entity.normalized_form}'"


def main():
    """
    Example usage demonstrating Article 14 processing as specified in the whitepaper.
    """
    # Initialize the knowledge base builder
    kb_builder = KnowledgeBaseBuilder()
    
    # Article 14 text from the whitepaper
    article_14_text = """Article 14 (Human Oversight): High-risk AI systems shall be designed and developed in such a way that they can be effectively overseen by natural persons during the period in which the AI system is in use."""
    
    print("AI Rosetta Stone - Symbolic Knowledge Base Builder")
    print("=" * 60)
    print(f"\nProcessing Article 14:")
    print(f"Input: {article_14_text}")
    
    # Process the article
    result = kb_builder.process_article(article_14_text)
    
    print(f"\nExtracted Entities ({len(result['entities'])}):")
    for i, entity in enumerate(result['entities'], 1):
        print(f"  {i}. {entity.text} -> {entity.normalized_form} ({entity.entity_type.value})")
    
    print(f"\nExtracted Relationships ({len(result['relationships'])}):")
    for i, rel in enumerate(result['relationships'], 1):
        print(f"  {i}. {rel.subject.normalized_form} --{rel.relation_type.value}--> {rel.object.normalized_form}")
    
    print(f"\nGenerated Logical Predicates ({len(result['predicates'])}):")
    for i, predicate in enumerate(result['predicates'], 1):
        print(f"  {i}. {predicate}")
    
    # Expected output format from whitepaper
    print(f"\nExpected Output Format (as Python list):")
    expected_predicates = [
        "requires(system_type='high-risk', component='human_oversight')",
        "is_a(human_oversight, 'effective')",
        "scope(human_oversight, period='in_use')"
    ]
    
    for predicate in expected_predicates:
        print(f"  \"{predicate}\"")
    
    return result


if __name__ == "__main__":
    main()