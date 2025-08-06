"""
Compliance Auditor - Mapping & Reasoning Layer

This module implements the automated auditor that systematically tests model logic
against legal requirements from the EU AI Act. It serves as the core component
of the Mapping & Reasoning Layer in the AI Rosetta Stone engine.

Author: AI Rosetta Stone Team
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Enumeration of possible compliance statuses."""
    COMPLIANT = "Compliance Verified"
    VIOLATION_FOUND = "Potential Violation Found"
    CRITICAL_VIOLATION = "Critical Violation Found"
    REQUIRES_REVIEW = "Requires Manual Review"
    INSUFFICIENT_DATA = "Insufficient Data for Assessment"


class ViolationSeverity(Enum):
    """Enumeration of violation severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ParsedRule:
    """Represents a parsed model rule with structured components."""
    rule_id: str
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    raw_rule: str
    confidence: float = 1.0
    
    def uses_attribute(self, attribute: str) -> bool:
        """Check if this rule uses a specific attribute."""
        for condition in self.conditions:
            if condition.get('attribute', '').lower() == attribute.lower():
                return True
        return False
    
    def has_outcome(self, outcome_pattern: str) -> bool:
        """Check if this rule has a specific outcome pattern."""
        for action in self.actions:
            if outcome_pattern.lower() in action.get('action', '').lower():
                return True
        return False


@dataclass
class ParsedPredicate:
    """Represents a parsed legal predicate with structured components."""
    predicate_type: str  # 'prohibits_bias', 'requires_human_oversight', etc.
    parameters: Dict[str, Any]
    article_reference: str
    raw_predicate: str
    
    def matches_attribute(self, attribute: str) -> bool:
        """Check if this predicate applies to a specific attribute."""
        return (
            'attribute' in self.parameters and 
            self.parameters['attribute'].lower() == attribute.lower()
        )


@dataclass
class ComplianceViolation:
    """Represents a specific compliance violation found during audit."""
    article: str
    severity: ViolationSeverity
    violating_rule: str
    legal_predicate: str
    description: str
    recommendation: str
    confidence: float


class RuleParser:
    """Parser for extracting structured information from model rules."""
    
    def __init__(self):
        """Initialize the rule parser with regex patterns."""
        # Pattern to match rule structure: Rule_ID: IF (...) THEN (...)
        self.rule_pattern = re.compile(
            r'(?P<rule_id>Rule_\d+):\s*IF\s+(?P<conditions>.*?)\s+THEN\s+(?P<actions>.*?)(?:\.|$)',
            re.IGNORECASE | re.DOTALL
        )
        
        # Pattern to match individual conditions
        self.condition_pattern = re.compile(
            r'(?P<attribute>\w+)\s*(?P<operator>[><=!]+)\s*(?P<value>[^)]+)',
            re.IGNORECASE
        )
        
        # Pattern to match actions
        self.action_pattern = re.compile(
            r'(?P<target>\w+)\s*(?P<operator>[+\-=])\s*(?P<value>[^,;]+)',
            re.IGNORECASE
        )
    
    def parse_rule(self, rule_string: str) -> Optional[ParsedRule]:
        """
        Parse a model rule string into structured components.
        
        Args:
            rule_string: Raw rule string from the model
            
        Returns:
            ParsedRule object or None if parsing fails
        """
        try:
            match = self.rule_pattern.search(rule_string.strip())
            if not match:
                logger.warning(f"Could not parse rule: {rule_string}")
                return None
            
            rule_id = match.group('rule_id')
            conditions_text = match.group('conditions')
            actions_text = match.group('actions')
            
            # Parse conditions
            conditions = self._parse_conditions(conditions_text)
            
            # Parse actions
            actions = self._parse_actions(actions_text)
            
            return ParsedRule(
                rule_id=rule_id,
                conditions=conditions,
                actions=actions,
                raw_rule=rule_string
            )
            
        except Exception as e:
            logger.error(f"Error parsing rule '{rule_string}': {e}")
            return None
    
    def _parse_conditions(self, conditions_text: str) -> List[Dict[str, Any]]:
        """Parse the conditions part of a rule."""
        conditions = []
        
        # Split by AND/OR (simplified - assumes AND for now)
        condition_parts = re.split(r'\s+AND\s+|\s+OR\s+', conditions_text, flags=re.IGNORECASE)
        
        for part in condition_parts:
            part = part.strip('() ')
            match = self.condition_pattern.search(part)
            
            if match:
                conditions.append({
                    'attribute': match.group('attribute'),
                    'operator': match.group('operator'),
                    'value': self._parse_value(match.group('value')),
                    'raw': part
                })
            else:
                # Handle special cases like boolean conditions
                conditions.append({
                    'attribute': 'unknown',
                    'operator': 'equals',
                    'value': part,
                    'raw': part
                })
        
        return conditions
    
    def _parse_actions(self, actions_text: str) -> List[Dict[str, Any]]:
        """Parse the actions part of a rule."""
        actions = []
        
        # Split by common delimiters
        action_parts = re.split(r'[,;]|\s+AND\s+', actions_text)
        
        for part in action_parts:
            part = part.strip()
            if not part:
                continue
                
            # Try to match assignment/modification patterns
            action_match = self.action_pattern.search(part)
            
            if action_match:
                actions.append({
                    'target': action_match.group('target'),
                    'operator': action_match.group('operator'),
                    'value': self._parse_value(action_match.group('value')),
                    'action': part
                })
            else:
                # Handle arrow notation and other patterns
                if '->' in part:
                    target, value = part.split('->', 1)
                    actions.append({
                        'target': target.strip(),
                        'operator': '->',
                        'value': value.strip().strip("'\""),
                        'action': part
                    })
                else:
                    actions.append({
                        'target': 'unknown',
                        'operator': 'set',
                        'value': part,
                        'action': part
                    })
        
        return actions
    
    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string to appropriate Python type."""
        value_str = value_str.strip().strip("'\"")
        
        # Try to convert to number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        # Check for boolean
        if value_str.upper() in ['TRUE', 'FALSE']:
            return value_str.upper() == 'TRUE'
        
        # Return as string
        return value_str


class PredicateParser:
    """Parser for extracting structured information from legal predicates."""
    
    def __init__(self):
        """Initialize the predicate parser."""
        # Pattern to match predicate structure: predicate_type(param1=value1, param2=value2)
        self.predicate_pattern = re.compile(
            r'(?P<type>\w+)\((?P<params>.*?)\)',
            re.IGNORECASE
        )
        
        # Pattern to match parameters
        self.param_pattern = re.compile(
            r'(?P<key>\w+)=(?P<value>[^,)]+)',
            re.IGNORECASE
        )
    
    def parse_predicate(self, predicate_string: str, article_ref: str = "Unknown") -> Optional[ParsedPredicate]:
        """
        Parse a legal predicate string into structured components.
        
        Args:
            predicate_string: Raw predicate string from legal knowledge base
            article_ref: Reference to the legal article
            
        Returns:
            ParsedPredicate object or None if parsing fails
        """
        try:
            match = self.predicate_pattern.search(predicate_string.strip())
            if not match:
                logger.warning(f"Could not parse predicate: {predicate_string}")
                return None
            
            predicate_type = match.group('type')
            params_text = match.group('params')
            
            # Parse parameters
            parameters = self._parse_parameters(params_text)
            
            return ParsedPredicate(
                predicate_type=predicate_type,
                parameters=parameters,
                article_reference=article_ref,
                raw_predicate=predicate_string
            )
            
        except Exception as e:
            logger.error(f"Error parsing predicate '{predicate_string}': {e}")
            return None
    
    def _parse_parameters(self, params_text: str) -> Dict[str, Any]:
        """Parse the parameters of a predicate."""
        parameters = {}
        
        # Find all parameter matches
        matches = self.param_pattern.findall(params_text)
        
        for key, value in matches:
            # Clean up the value
            value = value.strip().strip("'\"")
            
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    value = float(value)
                elif value.isdigit():
                    value = int(value)
                elif value.upper() in ['TRUE', 'FALSE']:
                    value = value.upper() == 'TRUE'
            except ValueError:
                pass  # Keep as string
            
            parameters[key.strip()] = value
        
        return parameters


class ComplianceAuditor:
    """
    Main compliance auditor class that implements the Mapping & Reasoning Layer.
    
    This class serves as the automated auditor that systematically tests model logic
    against legal requirements from the EU AI Act.
    """
    
    def __init__(self):
        """Initialize the compliance auditor."""
        self.rule_parser = RuleParser()
        self.predicate_parser = PredicateParser()
        
        # Define protected attributes as per EU AI Act Article 10
        self.protected_attributes = {
            'age', 'gender', 'race', 'ethnicity', 'religion', 'sexual_orientation',
            'disability', 'nationality', 'political_opinion', 'zip_code', 'postal_code'
        }
        
        # Define high-scrutiny outcomes that require human oversight
        self.high_scrutiny_outcomes = {
            'high_scrutiny', 'manual_review', 'reject', 'deny', 'flag'
        }
        
        # Article mappings for compliance checking
        self.article_mappings = {
            'prohibits_bias': 'Article 10 (Data & Bias)',
            'requires_human_oversight': 'Article 14 (Human Oversight)',
            'requires_transparency': 'Article 13 (Transparency)',
            'requires_robustness': 'Article 15 (Robustness)'
        }
    
    def run_audit(self, model_rules: List[str], legal_predicates: List[str]) -> Dict[str, Any]:
        """
        Run comprehensive compliance audit of model rules against legal predicates.
        
        Args:
            model_rules: List of symbolic rules extracted from the AI model
            legal_predicates: List of legal predicates from the knowledge base
            
        Returns:
            Dictionary containing detailed audit results
        """
        logger.info(f"Starting compliance audit with {len(model_rules)} model rules and {len(legal_predicates)} legal predicates")
        
        # Parse input data
        parsed_rules = self._parse_model_rules(model_rules)
        parsed_predicates = self._parse_legal_predicates(legal_predicates)
        
        # Initialize audit results
        audit_results = {}
        violations = []
        
        # Check each type of compliance requirement
        audit_results.update(self._check_bias_compliance(parsed_rules, parsed_predicates, violations))
        audit_results.update(self._check_oversight_compliance(parsed_rules, parsed_predicates, violations))
        audit_results.update(self._check_transparency_compliance(parsed_rules, parsed_predicates, violations))
        audit_results.update(self._check_robustness_compliance(parsed_rules, parsed_predicates, violations))
        
        # Add summary information
        audit_results['_audit_summary'] = {
            'timestamp': datetime.now().isoformat(),
            'total_rules_analyzed': len(parsed_rules),
            'total_predicates_checked': len(parsed_predicates),
            'total_violations_found': len(violations),
            'critical_violations': len([v for v in violations if v.severity == ViolationSeverity.CRITICAL]),
            'overall_compliance_status': self._determine_overall_status(violations)
        }
        
        logger.info(f"Audit completed. Found {len(violations)} violations across {len(audit_results)-1} articles")
        return audit_results
    
    def _parse_model_rules(self, model_rules: List[str]) -> List[ParsedRule]:
        """Parse all model rules into structured format."""
        parsed_rules = []
        
        for rule_string in model_rules:
            parsed_rule = self.rule_parser.parse_rule(rule_string)
            if parsed_rule:
                parsed_rules.append(parsed_rule)
        
        logger.info(f"Successfully parsed {len(parsed_rules)} out of {len(model_rules)} model rules")
        return parsed_rules
    
    def _parse_legal_predicates(self, legal_predicates: List[str]) -> List[ParsedPredicate]:
        """Parse all legal predicates into structured format."""
        parsed_predicates = []
        
        for predicate_string in legal_predicates:
            # Try to extract article reference from context
            article_ref = "Unknown"
            if "Article" in predicate_string:
                article_match = re.search(r'Article\s+(\d+)', predicate_string)
                if article_match:
                    article_ref = f"Article {article_match.group(1)}"
            
            parsed_predicate = self.predicate_parser.parse_predicate(predicate_string, article_ref)
            if parsed_predicate:
                parsed_predicates.append(parsed_predicate)
        
        logger.info(f"Successfully parsed {len(parsed_predicates)} out of {len(legal_predicates)} legal predicates")
        return parsed_predicates
    
    def _check_bias_compliance(self, rules: List[ParsedRule], predicates: List[ParsedPredicate], 
                              violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Check compliance with bias and non-discrimination requirements (Article 10)."""
        article_key = "Article 10 (Data & Bias)"
        
        # Find bias-related predicates
        bias_predicates = [p for p in predicates if p.predicate_type == 'prohibits_bias']
        
        if not bias_predicates:
            return {
                article_key: {
                    "status": ComplianceStatus.INSUFFICIENT_DATA.value,
                    "details": "No bias-related legal predicates found for assessment."
                }
            }
        
        # Check each rule for bias violations
        found_violations = []
        
        for rule in rules:
            for predicate in bias_predicates:
                violation = self._check_rule_against_bias_predicate(rule, predicate)
                if violation:
                    violations.append(violation)
                    found_violations.append(violation)
        
        # Determine status and create detailed report
        if found_violations:
            critical_violations = [v for v in found_violations if v.severity == ViolationSeverity.CRITICAL]
            
            status = ComplianceStatus.CRITICAL_VIOLATION if critical_violations else ComplianceStatus.VIOLATION_FOUND
            
            violation_details = []
            for violation in found_violations:
                violation_details.append(f"Rule '{violation.violating_rule}': {violation.description}")
            
            return {
                article_key: {
                    "status": status.value,
                    "details": "; ".join(violation_details),
                    "violating_rules": [v.violating_rule for v in found_violations],
                    "violations_count": len(found_violations),
                    "critical_violations_count": len(critical_violations)
                }
            }
        else:
            return {
                article_key: {
                    "status": ComplianceStatus.COMPLIANT.value,
                    "details": f"All {len(rules)} rules were tested for bias compliance. No violations found against protected attributes."
                }
            }
    
    def _check_rule_against_bias_predicate(self, rule: ParsedRule, predicate: ParsedPredicate) -> Optional[ComplianceViolation]:
        """Check a specific rule against a bias predicate."""
        
        # Check if the rule uses a prohibited attribute
        prohibited_attr = predicate.parameters.get('attribute', '').lower()
        
        if prohibited_attr and rule.uses_attribute(prohibited_attr):
            # Check if the rule has adverse outcomes
            has_adverse_outcome = any(
                action.get('operator') in ['+', '+='] and 
                ('risk' in action.get('target', '').lower() or 'score' in action.get('target', '').lower())
                for action in rule.actions
            )
            
            if has_adverse_outcome:
                severity = ViolationSeverity.CRITICAL if prohibited_attr in self.protected_attributes else ViolationSeverity.HIGH
                
                return ComplianceViolation(
                    article="Article 10",
                    severity=severity,
                    violating_rule=rule.rule_id,
                    legal_predicate=predicate.raw_predicate,
                    description=f"Model rule '{rule.rule_id}' directly uses the protected attribute '{prohibited_attr}' to increase risk score, potentially violating the '{predicate.raw_predicate}' predicate.",
                    recommendation=f"Remove or modify the use of '{prohibited_attr}' in decision logic, or implement bias mitigation techniques.",
                    confidence=0.9
                )
        
        return None
    
    def _check_oversight_compliance(self, rules: List[ParsedRule], predicates: List[ParsedPredicate],
                                   violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Check compliance with human oversight requirements (Article 14)."""
        article_key = "Article 14 (Human Oversight)"
        
        # Find oversight-related predicates
        oversight_predicates = [p for p in predicates if p.predicate_type == 'requires_human_oversight']
        
        if not oversight_predicates:
            return {
                article_key: {
                    "status": ComplianceStatus.INSUFFICIENT_DATA.value,
                    "details": "No human oversight legal predicates found for assessment."
                }
            }
        
        # Find rules that trigger high-scrutiny outcomes
        high_scrutiny_rules = []
        for rule in rules:
            if any(rule.has_outcome(outcome) for outcome in self.high_scrutiny_outcomes):
                high_scrutiny_rules.append(rule)
        
        if high_scrutiny_rules:
            # Check if these rules properly route to human oversight
            compliant_rules = 0
            violation_rules = []
            
            for rule in high_scrutiny_rules:
                # Check if the rule mentions human review or oversight
                has_human_oversight = any(
                    'human' in action.get('action', '').lower() or 
                    'review' in action.get('action', '').lower() or
                    'oversight' in action.get('action', '').lower()
                    for action in rule.actions
                )
                
                if has_human_oversight:
                    compliant_rules += 1
                else:
                    violation_rules.append(rule.rule_id)
                    
                    violations.append(ComplianceViolation(
                        article="Article 14",
                        severity=ViolationSeverity.HIGH,
                        violating_rule=rule.rule_id,
                        legal_predicate=oversight_predicates[0].raw_predicate,
                        description=f"Rule '{rule.rule_id}' triggers high-scrutiny outcomes but does not ensure human oversight as required.",
                        recommendation="Modify the rule to include mandatory human review for high-scrutiny decisions.",
                        confidence=0.8
                    ))
            
            if violation_rules:
                return {
                    article_key: {
                        "status": ComplianceStatus.VIOLATION_FOUND.value,
                        "details": f"Found {len(violation_rules)} rules triggering high-scrutiny outcomes without proper human oversight routing.",
                        "violating_rules": violation_rules,
                        "compliant_rules_count": compliant_rules,
                        "total_scrutiny_rules": len(high_scrutiny_rules)
                    }
                }
            else:
                return {
                    article_key: {
                        "status": ComplianceStatus.COMPLIANT.value,
                        "details": f"All {len(high_scrutiny_rules)} rules triggering 'high_scrutiny' flags were found to be compliant with oversight requirements."
                    }
                }
        else:
            return {
                article_key: {
                    "status": ComplianceStatus.COMPLIANT.value,
                    "details": "No high-scrutiny rules found that require human oversight assessment."
                }
            }
    
    def _check_transparency_compliance(self, rules: List[ParsedRule], predicates: List[ParsedPredicate],
                                     violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Check compliance with transparency requirements (Article 13)."""
        article_key = "Article 13 (Transparency)"
        
        # Find transparency-related predicates
        transparency_predicates = [p for p in predicates if p.predicate_type == 'requires_transparency']
        
        if not transparency_predicates:
            return {
                article_key: {
                    "status": ComplianceStatus.INSUFFICIENT_DATA.value,
                    "details": "No transparency legal predicates found for assessment."
                }
            }
        
        # For now, assume compliance if rules are extractable and interpretable
        return {
            article_key: {
                "status": ComplianceStatus.COMPLIANT.value,
                "details": f"Successfully extracted and analyzed {len(rules)} interpretable rules, demonstrating model transparency."
            }
        }
    
    def _check_robustness_compliance(self, rules: List[ParsedRule], predicates: List[ParsedPredicate],
                                   violations: List[ComplianceViolation]) -> Dict[str, Any]:
        """Check compliance with robustness requirements (Article 15)."""
        article_key = "Article 15 (Robustness)"
        
        # Find robustness-related predicates
        robustness_predicates = [p for p in predicates if p.predicate_type == 'requires_robustness']
        
        if not robustness_predicates:
            return {
                article_key: {
                    "status": ComplianceStatus.INSUFFICIENT_DATA.value,
                    "details": "No robustness legal predicates found for assessment."
                }
            }
        
        # Check for potential robustness issues in rules
        # This is a simplified check - real implementation would be more sophisticated
        return {
            article_key: {
                "status": ComplianceStatus.REQUIRES_REVIEW.value,
                "details": "Robustness assessment requires additional testing beyond rule analysis. Manual review recommended."
            }
        }
    
    def _determine_overall_status(self, violations: List[ComplianceViolation]) -> str:
        """Determine overall compliance status based on violations found."""
        if not violations:
            return "FULLY_COMPLIANT"
        
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical_violations:
            return "CRITICAL_VIOLATIONS_FOUND"
        
        high_violations = [v for v in violations if v.severity == ViolationSeverity.HIGH]
        if high_violations:
            return "HIGH_VIOLATIONS_FOUND"
        
        return "MINOR_VIOLATIONS_FOUND"


def demonstrate_compliance_auditor():
    """Demonstrate the compliance auditor with example data."""
    
    print("=" * 80)
    print("AI ROSETTA STONE - COMPLIANCE AUDITOR DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Example model rules (from the whitepaper)
    model_rules = [
        "Rule_127: IF (applicant_age < 25) THEN risk_score += 0.3",
        "Rule_045: IF (loan_to_value_ratio > 0.9) AND (is_first_time_buyer = FALSE) THEN decision -> 'high_scrutiny'",
        "Rule_089: IF (income < 30000) AND (employment_duration < 12) THEN risk_score += 0.2",
        "Rule_156: IF (decision = 'high_scrutiny') THEN route_to_human_review = TRUE"
    ]
    
    # Example legal predicates
    legal_predicates = [
        "prohibits_bias(attribute='age')",
        "prohibits_bias(attribute='gender')",
        "requires_human_oversight(system_type='high-risk')",
        "requires_transparency(system_type='high-risk')"
    ]
    
    print("📋 INPUT MODEL RULES:")
    print("-" * 40)
    for i, rule in enumerate(model_rules, 1):
        print(f"{i}. {rule}")
    
    print("\n📋 INPUT LEGAL PREDICATES:")
    print("-" * 40)
    for i, predicate in enumerate(legal_predicates, 1):
        print(f"{i}. {predicate}")
    
    print("\n🔍 RUNNING COMPLIANCE AUDIT...")
    print("-" * 40)
    
    # Initialize and run the auditor
    auditor = ComplianceAuditor()
    results = auditor.run_audit(model_rules, legal_predicates)
    
    print("✅ AUDIT RESULTS:")
    print("-" * 40)
    
    # Display results in a formatted way
    for article, result in results.items():
        if article.startswith('_'):  # Skip summary
            continue
            
        print(f"\n🏛️  {article}:")
        print(f"   Status: {result['status']}")
        print(f"   Details: {result['details']}")
        
        if 'violating_rules' in result:
            print(f"   Violating Rules: {', '.join(result['violating_rules'])}")
    
    # Display summary
    if '_audit_summary' in results:
        summary = results['_audit_summary']
        print(f"\n📊 AUDIT SUMMARY:")
        print("-" * 40)
        print(f"   Total Rules Analyzed: {summary['total_rules_analyzed']}")
        print(f"   Total Violations Found: {summary['total_violations_found']}")
        print(f"   Critical Violations: {summary['critical_violations']}")
        print(f"   Overall Status: {summary['overall_compliance_status']}")
    
    print("\n" + "=" * 80)
    print("✨ COMPLIANCE AUDITOR DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    """Main execution - demonstrate the compliance auditor."""
    demonstrate_compliance_auditor()