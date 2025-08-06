"""
AI Rosetta Stone - Symbolic Knowledge Base Builder (Demo Version)

This is a demonstration version that works without external dependencies
to showcase the core functionality of the Symbolic Knowledge Base component.

For production use, install spaCy: pip install spacy && python -m spacy download en_core_web_sm
"""

import re
from typing import List, Dict, Optional
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


class DemoKnowledgeBaseBuilder:
    """
    Demo version of the Knowledge Base Builder for immediate testing.
    
    This version uses regex patterns and rule-based extraction instead of
    advanced NLP libraries to demonstrate the core concept.
    """
    
    def __init__(self):
        """Initialize the demo knowledge base builder."""
        self._setup_patterns()
        
    def _setup_patterns(self):
        """Set up domain-specific patterns for entity and relationship extraction."""
        
        # System type patterns
        self.system_type_patterns = [
            (r"high-risk\s+AI\s+systems?", "high-risk"),
            (r"AI\s+systems?", "ai_system"),
            (r"artificial\s+intelligence\s+systems?", "ai_system"),
            (r"systems?", "system")
        ]
        
        # Component/requirement patterns
        self.component_patterns = [
            (r"human\s+oversight", "human_oversight"),
            (r"transparency", "transparency"),
            (r"robustness", "robustness"),
            (r"data\s+quality", "data_quality"),
            (r"accuracy", "accuracy"),
            (r"fairness", "fairness"),
            (r"accountability", "accountability"),
            (r"explainability", "explainability")
        ]
        
        # Person type patterns
        self.person_patterns = [
            (r"natural\s+persons?", "natural_persons"),
            (r"humans?", "humans"),
            (r"people", "people"),
            (r"individuals?", "individuals")
        ]
        
        # Time/scope patterns
        self.scope_patterns = [
            (r"during\s+the\s+period\s+in\s+which.*?in\s+use", "usage_period"),
            (r"in\s+use", "in_use"),
            (r"throughout\s+the\s+lifecycle", "lifecycle"),
            (r"at\s+all\s+times", "continuous"),
            (r"continuously", "continuous")
        ]
        
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from legal text using pattern matching."""
        entities = []
        text_lower = text.lower()
        
        # Extract system types
        for pattern, normalized in self.system_type_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    text=match.group(),
                    entity_type=EntityType.SYSTEM_TYPE,
                    normalized_form=normalized,
                    confidence=0.9
                ))
        
        # Extract components/requirements
        for pattern, normalized in self.component_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    text=match.group(),
                    entity_type=EntityType.COMPONENT,
                    normalized_form=normalized,
                    confidence=0.85
                ))
        
        # Extract person types
        for pattern, normalized in self.person_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    text=match.group(),
                    entity_type=EntityType.PERSON_TYPE,
                    normalized_form=normalized,
                    confidence=0.8
                ))
        
        # Extract scope/time patterns
        for pattern, normalized in self.scope_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    text=match.group(),
                    entity_type=EntityType.TIME_PERIOD,
                    normalized_form=normalized,
                    confidence=0.75
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
        """Extract relationships between entities using pattern matching."""
        relationships = []
        
        # Create lookups for entities by type
        system_entities = [e for e in entities if e.entity_type == EntityType.SYSTEM_TYPE]
        component_entities = [e for e in entities if e.entity_type == EntityType.COMPONENT]
        person_entities = [e for e in entities if e.entity_type == EntityType.PERSON_TYPE]
        scope_entities = [e for e in entities if e.entity_type == EntityType.TIME_PERIOD]
        
        # Pattern: "systems shall be designed...that they can be...overseen by persons"
        oversight_pattern = r"systems?\s+shall\s+be\s+designed.*?overseen\s+by\s+(.*?)(?:\s+during|\.|$)"
        matches = re.finditer(oversight_pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            if system_entities and person_entities:
                relationships.append(Relationship(
                    subject=system_entities[0],
                    relation_type=RelationType.OVERSEEN_BY,
                    object=person_entities[0],
                    confidence=0.9
                ))
        
        # Pattern: "systems shall be designed...that they can be effectively..."
        effective_pattern = r"effectively\s+(overseen|supervised|monitored)"
        if re.search(effective_pattern, text, re.IGNORECASE):
            for comp_entity in component_entities:
                if "oversight" in comp_entity.normalized_form:
                    # Create an "effective" attribute relationship
                    effective_entity = Entity(
                        text="effective",
                        entity_type=EntityType.ATTRIBUTE,
                        normalized_form="effective",
                        confidence=0.8
                    )
                    relationships.append(Relationship(
                        subject=comp_entity,
                        relation_type=RelationType.IS_A,
                        object=effective_entity,
                        confidence=0.85
                    ))
        
        # Pattern: Systems require human oversight
        if system_entities and component_entities:
            # Look for requirement patterns
            require_pattern = r"shall\s+be\s+designed.*?(?:that\s+they\s+can\s+be|to\s+be)"
            if re.search(require_pattern, text, re.IGNORECASE):
                for sys_entity in system_entities:
                    for comp_entity in component_entities:
                        if "oversight" in comp_entity.normalized_form:
                            relationships.append(Relationship(
                                subject=sys_entity,
                                relation_type=RelationType.REQUIRES,
                                object=comp_entity,
                                confidence=0.9
                            ))
        
        # Pattern: Scope relationships - "during the period in which...in use"
        scope_pattern = r"during\s+the\s+period.*?in\s+use"
        if re.search(scope_pattern, text, re.IGNORECASE):
            for comp_entity in component_entities:
                for scope_entity in scope_entities:
                    if scope_entity.normalized_form in ["usage_period", "in_use"]:
                        relationships.append(Relationship(
                            subject=comp_entity,
                            relation_type=RelationType.SCOPE,
                            object=scope_entity,
                            confidence=0.8
                        ))
        
        return relationships
    
    def convert_to_predicates(self, entities: List[Entity], relationships: List[Relationship]) -> List[str]:
        """Convert extracted entities and relationships into logical predicates."""
        predicates = []
        
        # Convert relationships to predicates
        for rel in relationships:
            if rel.relation_type == RelationType.REQUIRES:
                subject_param = self._entity_to_param(rel.subject)
                object_param = self._entity_to_param(rel.object)
                predicate = f"requires({subject_param}, {object_param})"
                predicates.append(predicate)
                
            elif rel.relation_type == RelationType.OVERSEEN_BY:
                subject_param = self._entity_to_param(rel.subject)
                object_param = self._entity_to_param(rel.object)
                predicate = f"overseen_by({subject_param}, {object_param})"
                predicates.append(predicate)
                
            elif rel.relation_type == RelationType.SCOPE:
                subject_param = self._entity_to_param(rel.subject)
                object_param = self._entity_to_param(rel.object)
                predicate = f"scope({subject_param}, {object_param})"
                predicates.append(predicate)
                
            elif rel.relation_type == RelationType.IS_A:
                if rel.subject.entity_type == EntityType.COMPONENT:
                    predicate = f"is_a({rel.subject.normalized_form}, '{rel.object.normalized_form}')"
                    predicates.append(predicate)
        
        return predicates
    
    def process_article(self, article_text: str) -> Dict:
        """Main method to process a legal article and extract logical predicates."""
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
    print("AI Rosetta Stone - Symbolic Knowledge Base Builder (Demo)")
    print("=" * 65)
    
    # Initialize the demo knowledge base builder
    kb_builder = DemoKnowledgeBaseBuilder()
    
    # Article 14 text from the whitepaper
    article_14_text = """Article 14 (Human Oversight): High-risk AI systems shall be designed and developed in such a way that they can be effectively overseen by natural persons during the period in which the AI system is in use."""
    
    print(f"\nProcessing Article 14:")
    print(f"Input: {article_14_text}")
    
    # Process the article
    result = kb_builder.process_article(article_14_text)
    
    print(f"\n1. Extracted Entities ({len(result['entities'])}):")
    for i, entity in enumerate(result['entities'], 1):
        print(f"   {i}. '{entity.text}' -> {entity.normalized_form} ({entity.entity_type.value})")
    
    print(f"\n2. Extracted Relationships ({len(result['relationships'])}):")
    for i, rel in enumerate(result['relationships'], 1):
        print(f"   {i}. {rel.subject.normalized_form} --{rel.relation_type.value}--> {rel.object.normalized_form}")
    
    print(f"\n3. Generated Logical Predicates ({len(result['predicates'])}):")
    for i, predicate in enumerate(result['predicates'], 1):
        print(f"   {i}. {predicate}")
    
    # Expected output format from whitepaper
    print(f"\n4. Expected Output Format (from whitepaper):")
    expected_predicates = [
        "requires(system_type='high-risk', component='human_oversight')",
        "is_a(human_oversight, 'effective')",
        "scope(human_oversight, period='in_use')"
    ]
    
    print("   Expected predicates:")
    for i, predicate in enumerate(expected_predicates, 1):
        print(f"   {i}. \"{predicate}\"")
    
    print(f"\n5. Analysis:")
    actual_predicates = result['predicates']
    
    # Check for matches with expected predicates
    matches = 0
    for expected in expected_predicates:
        for actual in actual_predicates:
            if _predicates_similar(expected, actual):
                matches += 1
                print(f"   ✓ Similar to expected: {expected}")
                break
        else:
            print(f"   ✗ Missing expected: {expected}")
    
    print(f"\n   Similarity score: {matches}/{len(expected_predicates)} expected predicates matched")
    
    if matches >= 2:
        print("   ✓ GOOD: Core functionality working - extracting key relationships!")
    else:
        print("   ⚠ PARTIAL: Some improvements needed for full compliance")
    
    print(f"\n6. Technical Notes:")
    print("   - This demo version uses rule-based pattern matching")
    print("   - Production version with spaCy would provide more accurate NER")
    print("   - Predicates can be extended with confidence scores and metadata")
    print("   - Ready for integration with reasoning engines (Prolog, OWL, etc.)")
    
    return result


def _predicates_similar(pred1: str, pred2: str) -> bool:
    """Check if two predicates are semantically similar."""
    # Simple similarity check - in production, this would be more sophisticated
    pred1_clean = pred1.lower().replace(" ", "").replace("'", "").replace('"', '')
    pred2_clean = pred2.lower().replace(" ", "").replace("'", "").replace('"', '')
    
    # Check for key components
    if "requires" in pred1_clean and "requires" in pred2_clean:
        return "high-risk" in pred1_clean and "human_oversight" in pred2_clean
    elif "is_a" in pred1_clean and "is_a" in pred2_clean:
        return "human_oversight" in pred1_clean and "effective" in pred2_clean
    elif "scope" in pred1_clean and "scope" in pred2_clean:
        return "human_oversight" in pred1_clean and ("in_use" in pred2_clean or "usage_period" in pred2_clean)
    
    return False


if __name__ == "__main__":
    main()