"""
AI Rosetta Stone - Mapping & Reasoning Layer
ComplianceAuditor Module

This module implements the automated auditor that takes symbolic rules extracted
from AI models and checks them against legal predicates from regulatory text.
Based on the AI Rosetta Stone whitepaper for EU AI Act compliance.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ComplianceStatus(Enum):
    """Enumeration for compliance status types."""
    COMPLIANT = "Compliance Verified"
    VIOLATION = "Potential Violation Found"
    REQUIRES_REVIEW = "Requires Manual Review"
    UNKNOWN = "Status Unknown"


@dataclass
class RuleCondition:
    """Represents a condition in a model rule."""
    attribute: str
    operator: str
    value: Any
    
    def __str__(self):
        return f"{self.attribute} {self.operator} {self.value}"


@dataclass
class ModelRule:
    """Represents a parsed model rule."""
    rule_id: str
    conditions: List[RuleCondition]
    action: str
    action_value: Any
    raw_rule: str
    
    def __str__(self):
        conditions_str = " AND ".join(str(cond) for cond in self.conditions)
        return f"{self.rule_id}: IF ({conditions_str}) THEN {self.action} {self.action_value}"


@dataclass
class LegalPredicate:
    """Represents a parsed legal predicate."""
    predicate_type: str
    attribute: Optional[str] = None
    system: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None
    raw_predicate: str = ""
    
    def __str__(self):
        if self.attribute:
            return f"{self.predicate_type}(attribute='{self.attribute}')"
        elif self.system:
            return f"{self.predicate_type}(system='{self.system}')"
        else:
            return f"{self.predicate_type}()"


class ComplianceAuditor:
    """
    The Mapping & Reasoning Layer component of the AI Rosetta Stone engine.
    
    This class serves as an automated auditor that takes symbolic rules extracted
    from AI models and checks them against logical predicates from legal text.
    """
    
    def __init__(self):
        """Initialize the ComplianceAuditor with predefined article mappings."""
        # Mapping of predicate types to EU AI Act articles
        self.article_mappings = {
            'prohibits_bias': 'Article 10 (Data & Bias)',
            'requires_human_oversight': 'Article 14 (Human Oversight)',
            'requires_transparency': 'Article 13 (Transparency)',
            'requires_robustness': 'Article 15 (Robustness)',
            'requires_data_quality': 'Article 10 (Data Quality)',
            'prohibits_discrimination': 'Article 10 (Non-discrimination)'
        }
        
        # Protected attributes that commonly trigger bias concerns
        self.protected_attributes = {
            'age', 'gender', 'race', 'ethnicity', 'religion', 'disability',
            'sexual_orientation', 'nationality', 'marital_status', 'zip_code',
            'postal_code', 'location', 'income_bracket'
        }
        
        # Actions that trigger oversight requirements
        self.oversight_triggers = {
            'high_scrutiny', 'manual_review', 'human_review', 'escalate',
            'flag_for_review', 'requires_approval'
        }
    
    def parse_model_rule(self, rule_string: str) -> Optional[ModelRule]:
        """
        Parse a model rule string into structured components.
        
        Args:
            rule_string: String like "Rule_127: IF (applicant_age < 25) THEN risk_score += 0.3"
            
        Returns:
            ModelRule object or None if parsing fails
        """
        # Pattern to match rule format: Rule_ID: IF (conditions) THEN action
        rule_pattern = r'(\w+):\s*IF\s*\((.*?)\)\s*THEN\s*(.*)'
        
        match = re.match(rule_pattern, rule_string.strip())
        if not match:
            return None
            
        rule_id = match.group(1)
        conditions_str = match.group(2)
        action_str = match.group(3)
        
        # Parse conditions
        conditions = []
        # Split by AND/OR (case insensitive)
        condition_parts = re.split(r'\s+AND\s+|\s+OR\s+', conditions_str, flags=re.IGNORECASE)
        
        for condition_part in condition_parts:
            condition_part = condition_part.strip().strip('()')
            # Match patterns like: attribute operator value
            condition_match = re.match(r'(\w+)\s*([<>=!]+)\s*(.+)', condition_part)
            if condition_match:
                attr = condition_match.group(1)
                op = condition_match.group(2)
                val = condition_match.group(3).strip("'\"").strip(')')
                
                # Try to convert numeric values
                try:
                    if '.' in str(val):
                        val = float(val)
                    else:
                        val = int(val)
                except ValueError:
                    # Keep as string if not numeric
                    pass
                
                conditions.append(RuleCondition(attr, op, val))
        
        # Parse action
        action_match = re.match(r'(\w+)\s*([\+\-\*\/]?=)\s*(.+)', action_str)
        if action_match:
            action = action_match.group(1)
            operator = action_match.group(2)
            value = action_match.group(3).strip("'\"")
            
            # Try to convert numeric values
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
                
            action_value = f"{operator} {value}" if operator != '=' else value
        else:
            # Simple assignment or flag
            parts = action_str.split('->')
            if len(parts) == 2:
                action = parts[0].strip()
                action_value = parts[1].strip().strip("'\"")
            else:
                action = action_str.strip()
                action_value = True
        
        return ModelRule(rule_id, conditions, action, action_value, rule_string)
    
    def parse_legal_predicate(self, predicate_string: str) -> Optional[LegalPredicate]:
        """
        Parse a legal predicate string into structured components.
        
        Args:
            predicate_string: String like "prohibits_bias(attribute='age')"
            
        Returns:
            LegalPredicate object or None if parsing fails
        """
        # Pattern to match predicate format: predicate_type(params)
        predicate_pattern = r'(\w+)\((.*?)\)'
        
        match = re.match(predicate_pattern, predicate_string.strip())
        if not match:
            return None
            
        predicate_type = match.group(1)
        params_str = match.group(2)
        
        # Parse parameters
        attribute = None
        system = None
        additional_params = {}
        
        if params_str:
            # Split parameters by comma
            param_parts = [p.strip() for p in params_str.split(',')]
            
            for param in param_parts:
                if '=' in param:
                    key, value = param.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    
                    if key == 'attribute':
                        attribute = value
                    elif key == 'system':
                        system = value
                    else:
                        additional_params[key] = value
        
        return LegalPredicate(predicate_type, attribute, system, additional_params, predicate_string)
    
    def check_bias_violation(self, rule: ModelRule, predicate: LegalPredicate) -> Tuple[ComplianceStatus, str]:
        """
        Check if a model rule violates bias prohibition predicates.
        
        Args:
            rule: Parsed model rule
            predicate: Parsed legal predicate
            
        Returns:
            Tuple of (status, details)
        """
        if predicate.predicate_type not in ['prohibits_bias', 'prohibits_discrimination']:
            return ComplianceStatus.UNKNOWN, "Not a bias-related predicate"
        
        # Check if rule uses protected attributes
        protected_attr_used = None
        for condition in rule.conditions:
            # Match both direct attribute names and variations (e.g., applicant_age matches age)
            attr_base = condition.attribute.lower().replace('applicant_', '').replace('_', '')
            predicate_attr = predicate.attribute.lower() if predicate.attribute else None
            
            if (condition.attribute.lower() in self.protected_attributes or 
                attr_base in self.protected_attributes):
                
                if predicate.attribute and (condition.attribute.lower() == predicate_attr or 
                                          attr_base == predicate_attr):
                    protected_attr_used = condition.attribute
                    break
                elif not predicate.attribute:  # General bias prohibition
                    protected_attr_used = condition.attribute
                    break
        
        if protected_attr_used:
            # Check if the rule has adverse impact
            adverse_actions = ['risk_score', 'penalty', 'rejection', 'denial', 'interest_rate']
            is_adverse = any(adverse in rule.action.lower() for adverse in adverse_actions)
            
            # Check for increasing penalties/risks or decreasing benefits
            has_adverse_effect = ('+=' in str(rule.action_value) or 
                                '>' in str(rule.action_value) or
                                'denied' in str(rule.action_value).lower() or
                                'reject' in str(rule.action_value).lower())
            
            if is_adverse and has_adverse_effect:
                return (ComplianceStatus.VIOLATION, 
                       f"Model rule '{rule.rule_id}' directly uses the protected attribute '{protected_attr_used}' "
                       f"to increase {rule.action}, potentially violating the '{predicate.raw_predicate}' predicate.")
        
        return (ComplianceStatus.COMPLIANT, 
               f"No direct use of protected attributes found in rule '{rule.rule_id}'.")
    
    def check_oversight_compliance(self, rule: ModelRule, predicate: LegalPredicate) -> Tuple[ComplianceStatus, str]:
        """
        Check if a model rule complies with human oversight requirements.
        
        Args:
            rule: Parsed model rule
            predicate: Parsed legal predicate
            
        Returns:
            Tuple of (status, details)
        """
        if predicate.predicate_type != 'requires_human_oversight':
            return ComplianceStatus.UNKNOWN, "Not an oversight-related predicate"
        
        # Check if rule triggers oversight mechanisms
        oversight_triggered = any(trigger in str(rule.action_value).lower() 
                                for trigger in self.oversight_triggers)
        
        if oversight_triggered:
            return (ComplianceStatus.COMPLIANT,
                   f"Rule '{rule.rule_id}' triggers '{rule.action_value}' flag, "
                   f"which is compliant with oversight requirements.")
        
        # Check if rule involves high-risk decisions
        high_risk_actions = ['decision', 'approval', 'rejection', 'classification']
        is_high_risk = any(risk_action in rule.action.lower() for risk_action in high_risk_actions)
        
        if is_high_risk:
            return (ComplianceStatus.VIOLATION,
                   f"Rule '{rule.rule_id}' makes high-risk decisions without triggering "
                   f"human oversight mechanisms as required by '{predicate.raw_predicate}'.")
        
        return (ComplianceStatus.COMPLIANT,
               f"Rule '{rule.rule_id}' does not involve high-risk decisions requiring oversight.")
    
    def check_transparency_compliance(self, rule: ModelRule, predicate: LegalPredicate) -> Tuple[ComplianceStatus, str]:
        """
        Check if a model rule complies with transparency requirements.
        
        Args:
            rule: Parsed model rule  
            predicate: Parsed legal predicate
            
        Returns:
            Tuple of (status, details)
        """
        if predicate.predicate_type != 'requires_transparency':
            return ComplianceStatus.UNKNOWN, "Not a transparency-related predicate"
        
        # For now, assume that having symbolic rules provides transparency
        return (ComplianceStatus.COMPLIANT,
               f"Rule '{rule.rule_id}' is expressed in symbolic form, "
               f"providing transparency as required by '{predicate.raw_predicate}'.")
    
    def run_audit(self, model_rules: List[str], legal_predicates: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Run compliance audit on model rules against legal predicates.
        
        Args:
            model_rules: List of rule strings from the neuro-symbolic bridge
            legal_predicates: List of predicate strings from the symbolic knowledge base
            
        Returns:
            Dictionary summarizing compliance findings by article
        """
        # Parse inputs
        parsed_rules = []
        for rule_str in model_rules:
            parsed_rule = self.parse_model_rule(rule_str)
            if parsed_rule:
                parsed_rules.append(parsed_rule)
        
        parsed_predicates = []
        for predicate_str in legal_predicates:
            parsed_predicate = self.parse_legal_predicate(predicate_str)
            if parsed_predicate:
                parsed_predicates.append(parsed_predicate)
        
        # Initialize results structure
        results = {}
        
        # Group predicates by article
        article_predicates = {}
        for predicate in parsed_predicates:
            article = self.article_mappings.get(predicate.predicate_type, 
                                              f"Unknown Article ({predicate.predicate_type})")
            if article not in article_predicates:
                article_predicates[article] = []
            article_predicates[article].append(predicate)
        
        # Check each article's predicates against all rules
        for article, predicates in article_predicates.items():
            article_violations = []
            article_compliant = []
            violating_rules = []
            
            for predicate in predicates:
                for rule in parsed_rules:
                    # Check compliance based on predicate type
                    if predicate.predicate_type in ['prohibits_bias', 'prohibits_discrimination']:
                        status, details = self.check_bias_violation(rule, predicate)
                    elif predicate.predicate_type == 'requires_human_oversight':
                        status, details = self.check_oversight_compliance(rule, predicate)
                    elif predicate.predicate_type == 'requires_transparency':
                        status, details = self.check_transparency_compliance(rule, predicate)
                    else:
                        status = ComplianceStatus.REQUIRES_REVIEW
                        details = f"Manual review required for predicate type '{predicate.predicate_type}'"
                    
                    if status == ComplianceStatus.VIOLATION:
                        article_violations.append(details)
                        violating_rules.append(rule.rule_id)
                    elif status == ComplianceStatus.COMPLIANT:
                        article_compliant.append(details)
            
            # Determine overall article status
            if article_violations:
                results[article] = {
                    "status": ComplianceStatus.VIOLATION.value,
                    "details": "; ".join(article_violations),
                    "violating_rule": violating_rules[0] if violating_rules else None
                }
            elif article_compliant:
                results[article] = {
                    "status": ComplianceStatus.COMPLIANT.value,
                    "details": f"All rules were found to be compliant with {article.lower()} requirements."
                }
            else:
                results[article] = {
                    "status": ComplianceStatus.REQUIRES_REVIEW.value,
                    "details": f"Manual review required for {article.lower()} compliance."
                }
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize the auditor
    auditor = ComplianceAuditor()
    
    # Example model rules from the whitepaper
    example_rules = [
        "Rule_127: IF (applicant_age < 25) THEN risk_score += 0.3",
        "Rule_045: IF (loan_to_value_ratio > 0.9) AND (is_first_time_buyer = FALSE) THEN decision -> 'high_scrutiny'",
        "Rule_089: IF (credit_score < 600) THEN approval_status = 'denied'"
    ]
    
    # Example legal predicates
    example_predicates = [
        "prohibits_bias(attribute='age')",
        "requires_human_oversight(system='credit_scoring')",
        "requires_transparency(system='loan_approval')"
    ]
    
    # Run the audit
    audit_results = auditor.run_audit(example_rules, example_predicates)
    
    # Display results
    print("=== AI Rosetta Stone Compliance Audit Results ===\n")
    for article, result in audit_results.items():
        print(f"{article}:")
        print(f"  Status: {result['status']}")
        print(f"  Details: {result['details']}")
        if 'violating_rule' in result and result['violating_rule']:
            print(f"  Violating Rule: {result['violating_rule']}")
        print()