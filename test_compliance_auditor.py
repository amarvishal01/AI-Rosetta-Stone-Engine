"""
Test suite for the AI Rosetta Stone ComplianceAuditor module.
Demonstrates various scenarios and validates functionality.
"""

import unittest
from compliance_auditor import ComplianceAuditor, ComplianceStatus


class TestComplianceAuditor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.auditor = ComplianceAuditor()
    
    def test_bias_violation_detection(self):
        """Test detection of bias violations using protected attributes."""
        # Test case from whitepaper - should detect violation
        model_rules = [
            "Rule_127: IF (applicant_age < 25) THEN risk_score += 0.3"
        ]
        legal_predicates = [
            "prohibits_bias(attribute='age')"
        ]
        
        results = self.auditor.run_audit(model_rules, legal_predicates)
        
        self.assertIn("Article 10 (Data & Bias)", results)
        self.assertEqual(results["Article 10 (Data & Bias)"]["status"], "Potential Violation Found")
        self.assertEqual(results["Article 10 (Data & Bias)"]["violating_rule"], "Rule_127")
        self.assertIn("applicant_age", results["Article 10 (Data & Bias)"]["details"])
    
    def test_oversight_compliance(self):
        """Test human oversight compliance detection."""
        # Rule that properly triggers oversight
        model_rules = [
            "Rule_045: IF (loan_to_value_ratio > 0.9) AND (is_first_time_buyer = FALSE) THEN decision -> 'high_scrutiny'"
        ]
        legal_predicates = [
            "requires_human_oversight(system='credit_scoring')"
        ]
        
        results = self.auditor.run_audit(model_rules, legal_predicates)
        
        self.assertIn("Article 14 (Human Oversight)", results)
        self.assertEqual(results["Article 14 (Human Oversight)"]["status"], "Compliance Verified")
    
    def test_oversight_violation(self):
        """Test detection of oversight violations."""
        # Rule that makes high-risk decisions without oversight
        model_rules = [
            "Rule_089: IF (credit_score < 600) THEN approval_status = 'denied'"
        ]
        legal_predicates = [
            "requires_human_oversight(system='credit_scoring')"
        ]
        
        results = self.auditor.run_audit(model_rules, legal_predicates)
        
        self.assertIn("Article 14 (Human Oversight)", results)
        self.assertEqual(results["Article 14 (Human Oversight)"]["status"], "Potential Violation Found")
        self.assertEqual(results["Article 14 (Human Oversight)"]["violating_rule"], "Rule_089")
    
    def test_transparency_compliance(self):
        """Test transparency compliance (symbolic rules provide transparency)."""
        model_rules = [
            "Rule_001: IF (income > 50000) THEN eligibility = 'approved'"
        ]
        legal_predicates = [
            "requires_transparency(system='loan_approval')"
        ]
        
        results = self.auditor.run_audit(model_rules, legal_predicates)
        
        self.assertIn("Article 13 (Transparency)", results)
        self.assertEqual(results["Article 13 (Transparency)"]["status"], "Compliance Verified")
    
    def test_multiple_violations(self):
        """Test detection of multiple violations across different articles."""
        model_rules = [
            "Rule_127: IF (applicant_age < 25) THEN risk_score += 0.3",  # Bias violation
            "Rule_089: IF (credit_score < 600) THEN approval_status = 'denied'",  # Oversight violation
            "Rule_200: IF (gender = 'female') THEN interest_rate += 0.5"  # Another bias violation
        ]
        legal_predicates = [
            "prohibits_bias(attribute='age')",
            "prohibits_bias(attribute='gender')",
            "requires_human_oversight(system='credit_scoring')"
        ]
        
        results = self.auditor.run_audit(model_rules, legal_predicates)
        
        # Should detect bias violations
        self.assertIn("Article 10 (Data & Bias)", results)
        self.assertEqual(results["Article 10 (Data & Bias)"]["status"], "Potential Violation Found")
        
        # Should detect oversight violation
        self.assertIn("Article 14 (Human Oversight)", results)
        self.assertEqual(results["Article 14 (Human Oversight)"]["status"], "Potential Violation Found")
    
    def test_compliant_rules(self):
        """Test rules that are fully compliant."""
        model_rules = [
            "Rule_001: IF (income > 50000) THEN eligibility = 'approved'",
            "Rule_002: IF (debt_to_income_ratio > 0.4) THEN decision -> 'manual_review'"
        ]
        legal_predicates = [
            "prohibits_bias(attribute='age')",
            "requires_human_oversight(system='credit_scoring')",
            "requires_transparency(system='loan_approval')"
        ]
        
        results = self.auditor.run_audit(model_rules, legal_predicates)
        
        # All articles should be compliant
        for article, result in results.items():
            self.assertEqual(result["status"], "Compliance Verified")
    
    def test_rule_parsing(self):
        """Test rule parsing functionality."""
        rule_string = "Rule_127: IF (applicant_age < 25) AND (income > 30000) THEN risk_score += 0.3"
        parsed_rule = self.auditor.parse_model_rule(rule_string)
        
        self.assertIsNotNone(parsed_rule)
        self.assertEqual(parsed_rule.rule_id, "Rule_127")
        self.assertEqual(len(parsed_rule.conditions), 2)
        self.assertEqual(parsed_rule.conditions[0].attribute, "applicant_age")
        self.assertEqual(parsed_rule.conditions[0].operator, "<")
        self.assertEqual(parsed_rule.conditions[0].value, 25)
        self.assertEqual(parsed_rule.action, "risk_score")
        self.assertEqual(parsed_rule.action_value, "+= 0.3")
    
    def test_predicate_parsing(self):
        """Test predicate parsing functionality."""
        predicate_string = "prohibits_bias(attribute='age')"
        parsed_predicate = self.auditor.parse_legal_predicate(predicate_string)
        
        self.assertIsNotNone(parsed_predicate)
        self.assertEqual(parsed_predicate.predicate_type, "prohibits_bias")
        self.assertEqual(parsed_predicate.attribute, "age")
        self.assertEqual(parsed_predicate.raw_predicate, predicate_string)


def run_comprehensive_demo():
    """Run a comprehensive demonstration of the ComplianceAuditor."""
    print("=== AI Rosetta Stone ComplianceAuditor Demonstration ===\n")
    
    auditor = ComplianceAuditor()
    
    # Scenario 1: Whitepaper Example (with corrected bias detection)
    print("Scenario 1: Whitepaper Example")
    print("-" * 40)
    
    model_rules_1 = [
        "Rule_127: IF (applicant_age < 25) THEN risk_score += 0.3",
        "Rule_045: IF (loan_to_value_ratio > 0.9) AND (is_first_time_buyer = FALSE) THEN decision -> 'high_scrutiny'"
    ]
    
    legal_predicates_1 = [
        "prohibits_bias(attribute='age')",
        "requires_human_oversight(system='credit_scoring')"
    ]
    
    results_1 = auditor.run_audit(model_rules_1, legal_predicates_1)
    
    for article, result in results_1.items():
        print(f"{article}:")
        print(f"  Status: {result['status']}")
        print(f"  Details: {result['details']}")
        if 'violating_rule' in result and result['violating_rule']:
            print(f"  Violating Rule: {result['violating_rule']}")
        print()
    
    # Scenario 2: Multiple Violations
    print("Scenario 2: Multiple Violations Detected")
    print("-" * 40)
    
    model_rules_2 = [
        "Rule_200: IF (gender = 'female') THEN interest_rate += 0.5",
        "Rule_201: IF (zip_code = '12345') THEN approval_status = 'denied'",
        "Rule_202: IF (credit_score < 500) THEN decision = 'reject'"
    ]
    
    legal_predicates_2 = [
        "prohibits_bias(attribute='gender')",
        "prohibits_bias(attribute='zip_code')",
        "requires_human_oversight(system='lending')"
    ]
    
    results_2 = auditor.run_audit(model_rules_2, legal_predicates_2)
    
    for article, result in results_2.items():
        print(f"{article}:")
        print(f"  Status: {result['status']}")
        print(f"  Details: {result['details']}")
        if 'violating_rule' in result and result['violating_rule']:
            print(f"  Violating Rule: {result['violating_rule']}")
        print()
    
    # Scenario 3: Fully Compliant System
    print("Scenario 3: Fully Compliant System")
    print("-" * 40)
    
    model_rules_3 = [
        "Rule_300: IF (income > 50000) THEN eligibility = 'approved'",
        "Rule_301: IF (debt_to_income_ratio > 0.4) THEN decision -> 'manual_review'",
        "Rule_302: IF (employment_years < 2) THEN decision -> 'human_review'"
    ]
    
    legal_predicates_3 = [
        "prohibits_bias(attribute='age')",
        "requires_human_oversight(system='lending')",
        "requires_transparency(system='approval')"
    ]
    
    results_3 = auditor.run_audit(model_rules_3, legal_predicates_3)
    
    for article, result in results_3.items():
        print(f"{article}:")
        print(f"  Status: {result['status']}")
        print(f"  Details: {result['details']}")
        print()


if __name__ == "__main__":
    # Run unit tests
    print("Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*60 + "\n")
    
    # Run comprehensive demo
    run_comprehensive_demo()