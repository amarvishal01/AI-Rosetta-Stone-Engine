"""
Basic Usage Example for AI Rosetta Stone Engine

This example demonstrates how to use the AI Rosetta Stone for compliance assessment.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rosetta_stone import (
    SymbolicKnowledgeBase,
    NeuroSymbolicBridge,
    MappingReasoningEngine,
    ComplianceReportGenerator
)


def create_sample_model():
    """Create a simple neural network for demonstration."""
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    return SimpleClassifier()


def main():
    """Run the basic usage example."""
    print("🚀 AI Rosetta Stone - Basic Usage Example")
    print("=" * 50)
    
    # Step 1: Create sample model and data
    print("\n📋 Step 1: Creating sample model and data...")
    model = create_sample_model()
    sample_data = torch.randn(1000, 10)  # 1000 samples, 10 features each
    print(f"   ✅ Created model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"   ✅ Generated {sample_data.shape[0]} sample data points")
    
    # Step 2: Initialize AI Rosetta Stone components
    print("\n🔧 Step 2: Initializing AI Rosetta Stone components...")
    knowledge_base = SymbolicKnowledgeBase()
    bridge = NeuroSymbolicBridge()
    reasoning_engine = MappingReasoningEngine(knowledge_base)
    report_generator = ComplianceReportGenerator()
    print("   ✅ All components initialized successfully")
    
    # Step 3: Extract symbolic rules from the model
    print("\n🔍 Step 3: Extracting symbolic rules from neural network...")
    try:
        extracted_rules = bridge.extract_rules(model, sample_data, method="decision_tree")
        print(f"   ✅ Successfully extracted {len(extracted_rules)} symbolic rules")
        
        # Display rule statistics
        report = bridge.generate_rule_report(extracted_rules)
        print(f"   📊 Average confidence: {report['average_confidence']:.2f}")
        print(f"   📊 High confidence rules: {report['high_confidence_rules']}")
        
    except Exception as e:
        print(f"   ❌ Rule extraction failed: {e}")
        return
    
    # Step 4: Assess compliance (with empty knowledge base for demo)
    print("\n⚖️  Step 4: Assessing regulatory compliance...")
    try:
        # Note: This will use an empty knowledge base since we haven't ingested legal documents
        assessment = reasoning_engine.assess_compliance(
            system_id="demo_classifier",
            model_rules=[],  # Empty for demo since we need to convert extracted rules
            system_type="high_risk"
        )
        
        print(f"   📊 Compliance Status: {assessment.overall_status.value}")
        print(f"   📊 Confidence Score: {assessment.confidence_score:.1%}")
        print(f"   📊 Violations Found: {len(assessment.violations)}")
        print(f"   📊 Recommendations: {len(assessment.recommendations)}")
        
    except Exception as e:
        print(f"   ❌ Compliance assessment failed: {e}")
        return
    
    # Step 5: Generate compliance report
    print("\n📄 Step 5: Generating compliance report...")
    try:
        output_path = Path("demo_compliance_report.html")
        report_generator.generate_compliance_report(
            assessment=assessment,
            output_path=output_path,
            format="html"
        )
        print(f"   ✅ Report generated successfully: {output_path}")
        
    except Exception as e:
        print(f"   ❌ Report generation failed: {e}")
        print("   ℹ️  This is expected in the demo environment")
    
    # Summary
    print("\n🎉 Demo completed successfully!")
    print("\n📋 Summary:")
    print(f"   • Model analyzed: {type(model).__name__}")
    print(f"   • Rules extracted: {len(extracted_rules) if 'extracted_rules' in locals() else 0}")
    print(f"   • Compliance status: {assessment.overall_status.value if 'assessment' in locals() else 'Unknown'}")
    
    print("\n💡 Next Steps:")
    print("   1. Ingest legal documents into the knowledge base")
    print("   2. Train models on real data for meaningful rule extraction")
    print("   3. Configure compliance thresholds for your use case")
    print("   4. Set up automated monitoring for production systems")


if __name__ == "__main__":
    main()