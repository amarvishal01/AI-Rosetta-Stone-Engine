"""
Enhanced Compliance Auditor Demonstration

This script demonstrates the compliance auditor with more comprehensive examples
that clearly show both violations and compliant behavior.
"""

from compliance_auditor import ComplianceAuditor
import json


def run_enhanced_demo():
    """Run enhanced demonstration showing various compliance scenarios."""
    
    print("=" * 80)
    print("AI ROSETTA STONE - ENHANCED COMPLIANCE AUDITOR DEMO")
    print("=" * 80)
    print()
    
    # More comprehensive model rules showing both violations and compliance
    model_rules = [
        # VIOLATION: Uses protected attribute 'age' to increase risk
        "Rule_127: IF (applicant_age < 25) THEN risk_score += 0.3",
        
        # COMPLIANT: Uses non-protected attributes for risk assessment
        "Rule_089: IF (income < 30000) AND (employment_duration < 12) THEN risk_score += 0.2",
        
        # VIOLATION: Triggers high scrutiny but no human oversight
        "Rule_045: IF (loan_to_value_ratio > 0.9) AND (is_first_time_buyer = FALSE) THEN decision -> 'high_scrutiny'",
        
        # COMPLIANT: Properly routes high scrutiny to human review
        "Rule_156: IF (decision = 'high_scrutiny') THEN route_to_human_review = TRUE",
        
        # VIOLATION: Uses another protected attribute
        "Rule_203: IF (zip_code IN ['90210', '10001']) THEN risk_score += 0.4",
        
        # COMPLIANT: General business rule
        "Rule_078: IF (credit_score > 750) THEN interest_rate -= 0.5"
    ]
    
    # Comprehensive legal predicates
    legal_predicates = [
        "prohibits_bias(attribute='age')",
        "prohibits_bias(attribute='zip_code')",
        "prohibits_bias(attribute='gender')",
        "requires_human_oversight(system_type='high-risk')",
        "requires_transparency(system_type='high-risk')",
        "requires_robustness(system_type='high-risk')"
    ]
    
    print("📋 MODEL RULES BEING ANALYZED:")
    print("-" * 50)
    for i, rule in enumerate(model_rules, 1):
        print(f"{i}. {rule}")
    
    print(f"\n📋 LEGAL PREDICATES (COMPLIANCE REQUIREMENTS):")
    print("-" * 50)
    for i, predicate in enumerate(legal_predicates, 1):
        print(f"{i}. {predicate}")
    
    print(f"\n🔍 RUNNING COMPREHENSIVE COMPLIANCE AUDIT...")
    print("-" * 50)
    
    # Run the audit
    auditor = ComplianceAuditor()
    results = auditor.run_audit(model_rules, legal_predicates)
    
    print("✅ DETAILED AUDIT RESULTS:")
    print("=" * 50)
    
    # Display results with enhanced formatting
    for article, result in results.items():
        if article.startswith('_'):
            continue
            
        status = result['status']
        status_icon = {
            "Compliance Verified": "✅",
            "Potential Violation Found": "⚠️",
            "Critical Violation Found": "❌",
            "Requires Manual Review": "🔍",
            "Insufficient Data for Assessment": "❓"
        }.get(status, "❓")
        
        print(f"\n{status_icon} {article}")
        print(f"   Status: {status}")
        print(f"   Details: {result['details']}")
        
        if 'violating_rules' in result and result['violating_rules']:
            print(f"   🚨 Violating Rules: {', '.join(result['violating_rules'])}")
        
        if 'violations_count' in result:
            print(f"   📊 Total Violations: {result['violations_count']}")
            
        if 'critical_violations_count' in result:
            print(f"   🔥 Critical Violations: {result['critical_violations_count']}")
    
    # Enhanced summary
    if '_audit_summary' in results:
        summary = results['_audit_summary']
        print(f"\n📊 COMPREHENSIVE AUDIT SUMMARY:")
        print("=" * 50)
        print(f"   📅 Timestamp: {summary['timestamp']}")
        print(f"   📋 Total Rules Analyzed: {summary['total_rules_analyzed']}")
        print(f"   🔍 Total Predicates Checked: {summary['total_predicates_checked']}")
        print(f"   ⚠️  Total Violations Found: {summary['total_violations_found']}")
        print(f"   🔥 Critical Violations: {summary['critical_violations']}")
        print(f"   🏆 Overall Status: {summary['overall_compliance_status']}")
    
    # Generate compliance recommendations
    print(f"\n💡 COMPLIANCE RECOMMENDATIONS:")
    print("=" * 50)
    
    violations_found = summary['total_violations_found'] if '_audit_summary' in results else 0
    
    if violations_found == 0:
        print("   ✅ Excellent! No compliance violations detected.")
        print("   ✅ Your AI system appears to be compliant with EU AI Act requirements.")
        print("   💡 Continue monitoring during deployment for ongoing compliance.")
    else:
        print(f"   🚨 Found {violations_found} compliance violations that need attention:")
        print("   📝 1. Review and modify rules that use protected attributes")
        print("   📝 2. Ensure high-scrutiny decisions include human oversight")
        print("   📝 3. Implement bias mitigation techniques")
        print("   📝 4. Re-run compliance audit after making changes")
        
    print(f"\n🎯 EXPECTED WHITEPAPER OUTPUT FORMAT:")
    print("=" * 50)
    
    # Show the expected format from the whitepaper
    expected_format = {
        "Article 10 (Data & Bias)": {
            "status": "Potential Violation Found",
            "details": "Model rule 'Rule_127' directly uses the protected attribute 'age' to increase risk score, potentially violating the 'prohibits_bias(attribute='age')' predicate.",
            "violating_rule": "Rule_127"
        },
        "Article 14 (Human Oversight)": {
            "status": "Compliance Verified", 
            "details": "All rules triggering 'high_scrutiny' flags were found to be compliant with oversight requirements."
        }
    }
    
    print(json.dumps(expected_format, indent=2))
    
    print(f"\n" + "=" * 80)
    print("✨ ENHANCED COMPLIANCE AUDITOR DEMONSTRATION COMPLETE")
    print("   This demonstrates the AI Rosetta Stone's capability to automatically")
    print("   audit AI model logic against EU AI Act requirements.")
    print("=" * 80)


if __name__ == "__main__":
    run_enhanced_demo()