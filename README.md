# AI Rosetta Stone - Mapping & Reasoning Layer

## ComplianceAuditor Module

This module implements the **Mapping & Reasoning Layer** component of the AI Rosetta Stone engine, as described in the technical whitepaper "De-risking AI Deployment through Neuro-Symbolic Regulatory Compliance."

### Overview

The `ComplianceAuditor` serves as an automated auditor that takes symbolic rules extracted from AI models (via the Neuro-Symbolic Bridge) and checks them against logical predicates derived from regulatory text (from the Symbolic Knowledge Base). It provides direct, verifiable mapping between AI model logic and EU AI Act compliance requirements.

### Key Features

- **Automated Compliance Checking**: Systematically tests model logic against legal requirements
- **Multi-Article Support**: Covers key EU AI Act articles including Data & Bias (Article 10), Human Oversight (Article 14), and Transparency (Article 13)
- **Bias Detection**: Identifies potential discrimination based on protected attributes
- **Oversight Validation**: Ensures high-risk decisions trigger appropriate human review mechanisms
- **Structured Reporting**: Generates machine-auditable and human-readable compliance reports

### Installation

No external dependencies required. The module uses only Python standard library components:

```python
# Clone or copy the compliance_auditor.py file to your project
from compliance_auditor import ComplianceAuditor
```

### Quick Start

```python
from compliance_auditor import ComplianceAuditor

# Initialize the auditor
auditor = ComplianceAuditor()

# Define model rules (from Neuro-Symbolic Bridge)
model_rules = [
    "Rule_127: IF (applicant_age < 25) THEN risk_score += 0.3",
    "Rule_045: IF (loan_to_value_ratio > 0.9) THEN decision -> 'high_scrutiny'"
]

# Define legal predicates (from Symbolic Knowledge Base)
legal_predicates = [
    "prohibits_bias(attribute='age')",
    "requires_human_oversight(system='credit_scoring')"
]

# Run compliance audit
results = auditor.run_audit(model_rules, legal_predicates)

# Display results
for article, result in results.items():
    print(f"{article}: {result['status']}")
    if result['status'] == "Potential Violation Found":
        print(f"  Violating Rule: {result['violating_rule']}")
```

### API Reference

#### ComplianceAuditor Class

##### `__init__()`
Initialize the ComplianceAuditor with predefined EU AI Act article mappings and protected attribute definitions.

##### `run_audit(model_rules: List[str], legal_predicates: List[str]) -> Dict[str, Dict[str, Any]]`

**Primary method for running compliance audits.**

**Parameters:**
- `model_rules`: List of symbolic rule strings extracted from the AI model
- `legal_predicates`: List of legal predicate strings from regulatory text

**Returns:**
Dictionary with compliance findings organized by EU AI Act article:
```python
{
    "Article 10 (Data & Bias)": {
        "status": "Potential Violation Found",
        "details": "Model rule 'Rule_127' directly uses the protected attribute 'applicant_age'...",
        "violating_rule": "Rule_127"
    },
    "Article 14 (Human Oversight)": {
        "status": "Compliance Verified",
        "details": "All rules were found to be compliant with oversight requirements."
    }
}
```

**Status Values:**
- `"Compliance Verified"`: All rules comply with the article requirements
- `"Potential Violation Found"`: One or more rules potentially violate requirements
- `"Requires Manual Review"`: Automated assessment inconclusive
- `"Status Unknown"`: Predicate type not recognized

### Rule Format Specification

#### Model Rules
Model rules should follow the format:
```
Rule_ID: IF (condition1 AND condition2) THEN action
```

**Examples:**
```python
"Rule_127: IF (applicant_age < 25) THEN risk_score += 0.3"
"Rule_045: IF (loan_to_value_ratio > 0.9) AND (is_first_time_buyer = FALSE) THEN decision -> 'high_scrutiny'"
"Rule_089: IF (credit_score < 600) THEN approval_status = 'denied'"
```

**Supported Operators:**
- Comparison: `<`, `>`, `<=`, `>=`, `==`, `!=`
- Assignment: `=`, `+=`, `-=`, `*=`, `/=`
- Arrow notation: `->` for state transitions

#### Legal Predicates
Legal predicates should follow the format:
```
predicate_type(parameter='value')
```

**Supported Predicate Types:**
- `prohibits_bias(attribute='age')` - Prohibits discrimination based on specified attribute
- `prohibits_discrimination(attribute='gender')` - General discrimination prohibition  
- `requires_human_oversight(system='credit_scoring')` - Mandates human review for high-risk decisions
- `requires_transparency(system='loan_approval')` - Requires explainable decision-making
- `requires_robustness(system='classification')` - Ensures system reliability
- `requires_data_quality(system='recommendation')` - Mandates data quality standards

### Protected Attributes

The system recognizes the following protected attributes for bias detection:
- `age`, `gender`, `race`, `ethnicity`, `religion`
- `disability`, `sexual_orientation`, `nationality` 
- `marital_status`, `zip_code`, `postal_code`
- `location`, `income_bracket`

### Example Scenarios

#### Scenario 1: Bias Violation Detection
```python
model_rules = [
    "Rule_200: IF (gender = 'female') THEN interest_rate += 0.5"
]
legal_predicates = [
    "prohibits_bias(attribute='gender')"
]

results = auditor.run_audit(model_rules, legal_predicates)
# Result: Potential Violation Found
```

#### Scenario 2: Oversight Compliance
```python
model_rules = [
    "Rule_301: IF (debt_to_income_ratio > 0.4) THEN decision -> 'manual_review'"
]
legal_predicates = [
    "requires_human_oversight(system='lending')"
]

results = auditor.run_audit(model_rules, legal_predicates)
# Result: Compliance Verified
```

#### Scenario 3: Multiple Article Assessment
```python
model_rules = [
    "Rule_127: IF (applicant_age < 25) THEN risk_score += 0.3",
    "Rule_045: IF (high_risk = TRUE) THEN decision -> 'human_review'",
    "Rule_300: IF (income > 50000) THEN eligibility = 'approved'"
]
legal_predicates = [
    "prohibits_bias(attribute='age')",
    "requires_human_oversight(system='credit_scoring')",
    "requires_transparency(system='loan_approval')"
]

results = auditor.run_audit(model_rules, legal_predicates)
# Results cover multiple articles with different compliance statuses
```

### Testing

Run the comprehensive test suite:

```bash
python3 test_compliance_auditor.py
```

The test suite includes:
- Unit tests for core functionality
- Integration tests for end-to-end workflows
- Demonstration scenarios showing various compliance outcomes

### Architecture Integration

This module represents the **Mapping & Reasoning Layer** in the AI Rosetta Stone architecture:

```
[Neural Network Model] → [Neuro-Symbolic Bridge] → [Mapping & Reasoning Layer] → [Compliance Report]
                                                           ↑
[Regulatory Text] → [Symbolic Knowledge Base] → [Legal Predicates]
```

The ComplianceAuditor:
1. Receives symbolic rules from the Neuro-Symbolic Bridge
2. Receives legal predicates from the Symbolic Knowledge Base  
3. Performs automated compliance checking
4. Generates structured compliance reports

### Compliance Coverage

#### EU AI Act Articles Supported:
- **Article 10 (Data & Bias)**: Detects discriminatory use of protected attributes
- **Article 13 (Transparency)**: Validates explainability through symbolic representation
- **Article 14 (Human Oversight)**: Ensures high-risk decisions trigger human review
- **Article 15 (Robustness)**: Framework for robustness assessment (extensible)

### Extensibility

The module is designed for extensibility:

1. **Add New Predicate Types**: Extend `article_mappings` and add corresponding check methods
2. **Custom Protected Attributes**: Modify `protected_attributes` set
3. **Additional Compliance Logic**: Implement new check methods following the pattern
4. **Enhanced Parsing**: Extend rule and predicate parsing for complex formats

### Limitations

- **Rule Format**: Currently supports IF-THEN format with basic operators
- **Predicate Complexity**: Limited to simple parameter-value predicates
- **Static Analysis**: Performs static rule analysis, not dynamic behavior assessment
- **Language Support**: English-language rules and predicates only

### Future Enhancements

- Support for more complex rule formats (nested conditions, temporal logic)
- Integration with model training pipelines for proactive compliance
- Real-time monitoring capabilities for deployed models
- Multi-language support for international regulations
- Advanced bias detection using statistical parity measures

### License

This implementation is based on the AI Rosetta Stone whitepaper and is provided as a reference implementation for educational and research purposes.

---

For questions or contributions, please refer to the technical whitepaper: "De-risking AI Deployment through Neuro-Symbolic Regulatory Compliance"
