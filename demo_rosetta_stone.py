#!/usr/bin/env python3
"""
AI Rosetta Stone: Comprehensive Demo
Demonstrates the Neuro-Symbolic Bridge for regulatory compliance

This script shows how to:
1. Load a trained PyTorch model
2. Extract symbolic rules using GBAPA
3. Map rules to input features
4. Generate compliance reports
5. Export results for regulatory auditing

Usage: python demo_rosetta_stone.py
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Import our Neuro-Symbolic Bridge
from neuro_symbolic_bridge import (
    NeuroSymbolicBridge, 
    SymbolicRule, 
    ComplianceReport,
    create_sample_model
)

def create_realistic_credit_model() -> tuple:
    """Create a more realistic credit scoring model with sample data."""
    
    print("🏗️  Creating realistic credit scoring model...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define feature names (credit scoring domain)
    feature_names = [
        'credit_score',      # 300-850 range
        'annual_income',     # In thousands
        'age',              # 18-80
        'employment_years',  # 0-40
        'debt_to_income',   # 0.0-1.0 ratio
        'loan_amount',      # In thousands
        'education_level',  # 1-5 scale
        'marital_status',   # 0=single, 1=married
        'num_dependents',   # 0-5
        'property_value'    # In thousands
    ]
    
    class_names = ['approved', 'rejected']
    
    # Create a more sophisticated model architecture
    class CreditScoringModel(nn.Module):
        def __init__(self, input_dim=10):
            super().__init__()
            self.feature_encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            self.risk_analyzer = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            self.decision_layer = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
                nn.Softmax(dim=1)
            )
        
        def forward(self, x):
            features = self.feature_encoder(x)
            risk_profile = self.risk_analyzer(features)
            decision = self.decision_layer(risk_profile)
            return decision
    
    model = CreditScoringModel()
    
    # Generate realistic sample data
    n_samples = 2000
    
    # Generate correlated features that mimic real credit data
    data = np.zeros((n_samples, len(feature_names)))
    
    # Credit score (primary factor)
    data[:, 0] = np.random.normal(650, 100, n_samples)  # credit_score
    data[:, 0] = np.clip(data[:, 0], 300, 850)
    
    # Income (correlated with credit score)
    base_income = 30 + (data[:, 0] - 300) * 0.1  # Base income correlation
    data[:, 1] = np.maximum(20, np.random.normal(base_income, 25, n_samples))  # annual_income
    
    # Age
    data[:, 2] = np.random.normal(35, 12, n_samples)  # age
    data[:, 2] = np.clip(data[:, 2], 18, 80)
    
    # Employment years (correlated with age)
    data[:, 3] = np.maximum(0, np.minimum(data[:, 2] - 18, 
                           np.random.normal((data[:, 2] - 18) * 0.6, 5, n_samples)))
    
    # Debt to income ratio (inversely correlated with credit score)
    data[:, 4] = np.maximum(0, np.minimum(1.0, 
                           np.random.normal(0.8 - (data[:, 0] - 300) / 1000, 0.2, n_samples)))
    
    # Loan amount
    data[:, 5] = np.random.lognormal(np.log(50), 0.8, n_samples)  # loan_amount
    
    # Education level (1-5)
    data[:, 6] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1])
    
    # Marital status (binary)
    data[:, 7] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    
    # Number of dependents
    data[:, 8] = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.08, 0.02])
    
    # Property value (correlated with income)
    data[:, 9] = np.maximum(0, np.random.normal(data[:, 1] * 3, data[:, 1] * 0.5, n_samples))
    
    # Normalize features for neural network
    X_raw = data.copy()
    
    # Simple normalization (in practice, use proper scaling)
    X_normalized = np.zeros_like(data)
    X_normalized[:, 0] = (data[:, 0] - 300) / 550  # credit_score
    X_normalized[:, 1] = data[:, 1] / 200  # annual_income
    X_normalized[:, 2] = (data[:, 2] - 18) / 62  # age
    X_normalized[:, 3] = data[:, 3] / 40  # employment_years
    X_normalized[:, 4] = data[:, 4]  # debt_to_income (already 0-1)
    X_normalized[:, 5] = data[:, 5] / 500  # loan_amount
    X_normalized[:, 6] = (data[:, 6] - 1) / 4  # education_level
    X_normalized[:, 7] = data[:, 7]  # marital_status (already binary)
    X_normalized[:, 8] = data[:, 8] / 5  # num_dependents
    X_normalized[:, 9] = data[:, 9] / 1000  # property_value
    
    X_tensor = torch.FloatTensor(X_normalized)
    
    # Create a simple decision rule for training labels
    # High credit score + low debt ratio + sufficient income = approved
    approval_score = (
        (data[:, 0] - 300) / 550 * 0.4 +  # credit_score weight
        (1 - data[:, 4]) * 0.3 +          # inverse debt_to_income weight
        np.minimum(data[:, 1] / 100, 1.0) * 0.2 +  # income weight (capped)
        (data[:, 3] / 40) * 0.1           # employment_years weight
    )
    
    # Add some noise and create binary labels
    approval_threshold = 0.6 + np.random.normal(0, 0.1, n_samples)
    y_labels = (approval_score > approval_threshold).astype(int)
    
    # Train the model briefly (simplified training)
    print("🎯 Training model on synthetic credit data...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    y_tensor = torch.LongTensor(y_labels)
    
    # Quick training loop
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
    
    model.eval()
    
    # Calculate final accuracy
    with torch.no_grad():
        predictions = model(X_tensor)
        predicted_classes = torch.argmax(predictions, dim=1)
        accuracy = (predicted_classes == y_tensor).float().mean()
        print(f"  Final model accuracy: {accuracy:.3f}")
    
    return model, X_tensor, X_raw, feature_names, class_names, y_tensor

def demonstrate_rule_extraction():
    """Main demonstration of the Neuro-Symbolic Bridge."""
    
    print("\n" + "="*60)
    print("🚀 AI ROSETTA STONE: NEURO-SYMBOLIC BRIDGE DEMO")
    print("="*60)
    
    # Step 1: Create realistic model and data
    model, X_tensor, X_raw, feature_names, class_names, y_true = create_realistic_credit_model()
    
    print(f"\n📊 Dataset Info:")
    print(f"  • Samples: {len(X_tensor)}")
    print(f"  • Features: {len(feature_names)}")
    print(f"  • Classes: {class_names}")
    
    # Step 2: Initialize the Neuro-Symbolic Bridge
    print(f"\n🌉 Initializing Neuro-Symbolic Bridge...")
    bridge = NeuroSymbolicBridge(
        model=model,
        feature_names=feature_names,
        class_names=class_names,
        device='cpu'
    )
    
    # Step 3: Extract rules using GBAPA
    print(f"\n🔍 Extracting symbolic rules using GBAPA...")
    print("  This may take a few moments...")
    
    # Use a subset for faster demonstration
    sample_size = 500
    X_sample = X_tensor[:sample_size]
    
    try:
        rules = bridge.extract_rules(
            X_sample=X_sample,
            n_clusters=6,
            min_coverage=0.08,
            confidence_threshold=0.6
        )
        print(f"  ✅ Extracted {len(rules)} neuron-level rules")
        
        # Step 4: Map rules to input features
        print(f"\n🗺️  Mapping rules to input features...")
        mapped_rules = bridge.map_to_input_features(X_sample)
        print(f"  ✅ Created {len(mapped_rules)} feature-level rules")
        
        # Step 5: Display extracted rules
        print(f"\n📋 EXTRACTED SYMBOLIC RULES:")
        print("-" * 50)
        
        for i, rule in enumerate(mapped_rules[:8]):  # Show first 8 rules
            print(f"\n{i+1}. {rule.to_readable_string()}")
            print(f"   📈 Confidence: {rule.confidence:.3f} | Coverage: {rule.coverage:.1%}")
        
        # Step 6: Generate compliance report
        print(f"\n🏛️  Generating EU AI Act compliance report...")
        protected_features = ['age', 'marital_status']  # Protected attributes under EU law
        
        report = bridge.generate_compliance_report(
            rules=mapped_rules,
            protected_features=protected_features,
            fairness_threshold=0.02
        )
        
        # Step 7: Display compliance report
        print(f"\n📊 REGULATORY COMPLIANCE REPORT")
        print("=" * 50)
        print(f"🎯 Model ID: {report.model_id}")
        print(f"📏 Total Rules Analyzed: {report.total_rules}")
        print(f"✅ Compliant Rules: {report.compliant_rules}")
        print(f"⚠️  Non-Compliant Rules: {report.non_compliant_rules}")
        print(f"🚨 Risk Level: {report.risk_level}")
        
        print(f"\n📜 EU AI Act Article Compliance:")
        for article, compliant in report.article_compliance.items():
            status = "✅ PASS" if compliant else "❌ FAIL"
            print(f"  • {article.replace('_', ' ')}: {status}")
        
        if report.recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Step 8: Show detailed compliance issues
        high_severity_issues = [
            finding for finding in report.detailed_findings 
            if finding['issues'] and any(issue['severity'] == 'HIGH' for issue in finding['issues'])
        ]
        
        if high_severity_issues:
            print(f"\n⚠️  HIGH SEVERITY COMPLIANCE ISSUES:")
            print("-" * 40)
            for finding in high_severity_issues[:3]:  # Show first 3
                print(f"\n🔴 Rule: {finding['rule_id']}")
                for issue in finding['issues']:
                    if issue['severity'] == 'HIGH':
                        print(f"   • {issue['type']}: {issue['description']}")
                        print(f"     Article: {issue['article']}")
        
        # Step 9: Export results
        print(f"\n💾 Exporting results...")
        
        # Export rules to JSON
        bridge.export_rules_to_json(mapped_rules, '/workspace/extracted_rules.json')
        
        # Export compliance report
        report_data = {
            'model_id': report.model_id,
            'summary': {
                'total_rules': report.total_rules,
                'compliant_rules': report.compliant_rules,
                'non_compliant_rules': report.non_compliant_rules,
                'risk_level': report.risk_level
            },
            'article_compliance': report.article_compliance,
            'recommendations': report.recommendations,
            'detailed_findings': report.detailed_findings
        }
        
        with open('/workspace/compliance_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"  ✅ Rules exported to: /workspace/extracted_rules.json")
        print(f"  ✅ Report exported to: /workspace/compliance_report.json")
        
        # Step 10: Create visualization
        create_compliance_visualization(report, mapped_rules)
        
        return bridge, mapped_rules, report
        
    except Exception as e:
        print(f"❌ Error during rule extraction: {str(e)}")
        print("This might be due to missing dependencies. Please install:")
        print("pip install captum scikit-learn torch")
        return None, None, None

def create_compliance_visualization(report: ComplianceReport, rules: List[SymbolicRule]):
    """Create visualization of compliance analysis."""
    
    print(f"\n📊 Creating compliance visualization...")
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AI Rosetta Stone: Compliance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Compliance Overview (Pie Chart)
        compliance_data = [report.compliant_rules, report.non_compliant_rules]
        compliance_labels = ['Compliant', 'Non-Compliant']
        colors = ['#2ecc71', '#e74c3c']
        
        ax1.pie(compliance_data, labels=compliance_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Rule Compliance Overview')
        
        # 2. Article Compliance (Bar Chart)
        articles = list(report.article_compliance.keys())
        article_status = [1 if report.article_compliance[art] else 0 for art in articles]
        article_labels = [art.replace('Article_', '').replace('_', '\n') for art in articles]
        
        bars = ax2.bar(article_labels, article_status, color=['#2ecc71' if x == 1 else '#e74c3c' for x in article_status])
        ax2.set_title('EU AI Act Article Compliance')
        ax2.set_ylabel('Compliance Status')
        ax2.set_ylim(0, 1.2)
        
        # Add text labels on bars
        for bar, status in zip(bars, article_status):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    'PASS' if status == 1 else 'FAIL',
                    ha='center', va='bottom', fontweight='bold')
        
        # 3. Rule Quality Distribution
        confidences = [rule.confidence for rule in rules]
        coverages = [rule.coverage for rule in rules]
        
        ax3.scatter(coverages, confidences, alpha=0.6, s=50)
        ax3.set_xlabel('Rule Coverage')
        ax3.set_ylabel('Rule Confidence')
        ax3.set_title('Rule Quality Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Add quality thresholds
        ax3.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Min Confidence')
        ax3.axvline(x=0.05, color='r', linestyle='--', alpha=0.5, label='Min Coverage')
        ax3.legend()
        
        # 4. Risk Level Indicator
        risk_colors = {'LOW': '#2ecc71', 'MEDIUM': '#f39c12', 'HIGH': '#e74c3c'}
        risk_color = risk_colors.get(report.risk_level, '#95a5a6')
        
        ax4.bar(['Risk Level'], [1], color=risk_color, width=0.5)
        ax4.set_title('Overall Risk Assessment')
        ax4.set_ylabel('Risk Level')
        ax4.set_ylim(0, 1.2)
        ax4.text(0, 0.5, report.risk_level, ha='center', va='center', 
                fontsize=20, fontweight='bold', color='white')
        
        # Remove ticks for cleaner look
        ax4.set_xticks([])
        ax4.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('/workspace/compliance_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Dashboard saved to: /workspace/compliance_dashboard.png")
        
        # Don't show the plot in background mode
        plt.close()
        
    except Exception as e:
        print(f"  ⚠️  Could not create visualization: {str(e)}")
        print("  This might be due to missing matplotlib. Install with: pip install matplotlib")

def analyze_rule_quality(rules: List[SymbolicRule]):
    """Analyze the quality and characteristics of extracted rules."""
    
    print(f"\n🔬 RULE QUALITY ANALYSIS")
    print("-" * 40)
    
    if not rules:
        print("No rules to analyze.")
        return
    
    confidences = [rule.confidence for rule in rules]
    coverages = [rule.coverage for rule in rules]
    
    print(f"📊 Statistical Summary:")
    print(f"  • Total Rules: {len(rules)}")
    print(f"  • Avg Confidence: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
    print(f"  • Avg Coverage: {np.mean(coverages):.3f} ± {np.std(coverages):.3f}")
    print(f"  • High Quality Rules (conf>0.8, cov>0.05): {sum(1 for r in rules if r.confidence > 0.8 and r.coverage > 0.05)}")
    
    # Find most important features across all rules
    feature_importance = {}
    for rule in rules:
        for condition in rule.conditions:
            feature = condition['feature']
            importance = condition.get('importance', 0)
            if feature in feature_importance:
                feature_importance[feature] += importance
            else:
                feature_importance[feature] = importance
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 Most Important Features:")
    for i, (feature, importance) in enumerate(sorted_features[:5], 1):
        print(f"  {i}. {feature}: {importance:.4f}")

def main():
    """Main execution function."""
    
    print("Starting AI Rosetta Stone demonstration...")
    
    # Run the main demonstration
    bridge, rules, report = demonstrate_rule_extraction()
    
    if rules and report:
        # Additional analysis
        analyze_rule_quality(rules)
        
        print(f"\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✨ Key Achievements:")
        print(f"  • Extracted {len(rules)} high-fidelity symbolic rules")
        print(f"  • Generated comprehensive compliance report")
        print(f"  • Risk level assessed as: {report.risk_level}")
        print(f"  • Results exported for regulatory review")
        print(f"\n📁 Output Files:")
        print(f"  • /workspace/extracted_rules.json")
        print(f"  • /workspace/compliance_report.json")
        print(f"  • /workspace/compliance_dashboard.png")
        
        print(f"\n🚀 Next Steps for Production Deployment:")
        print(f"  1. Integrate with your existing PyTorch models")
        print(f"  2. Configure protected attributes for your domain")
        print(f"  3. Set up automated compliance monitoring")
        print(f"  4. Establish audit trails for regulatory review")
        
    else:
        print(f"\n❌ Demonstration failed. Please check dependencies and try again.")
        print(f"Required packages: torch, captum, scikit-learn, numpy, pandas, matplotlib")

if __name__ == "__main__":
    main()