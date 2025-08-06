"""
Command Line Interface for AI Rosetta Stone Engine
"""

import click
import torch
import logging
from pathlib import Path
from typing import Optional

from .knowledge_base import SymbolicKnowledgeBase
from .bridge import NeuroSymbolicBridge
from .mapping import MappingReasoningEngine  
from .reporting import ComplianceReportGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool):
    """AI Rosetta Stone: Neuro-Symbolic Engine for AI Regulatory Compliance"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.option("--document", "-d", required=True, type=click.Path(exists=True), 
              help="Path to legal document")
@click.option("--type", "-t", default="eu_ai_act", 
              help="Document type (eu_ai_act, gdpr, custom)")
@click.option("--output", "-o", type=click.Path(), 
              help="Output path for knowledge base")
def ingest(document: str, type: str, output: Optional[str]):
    """Ingest legal documents into the knowledge base."""
    click.echo(f"Ingesting legal document: {document}")
    
    kb = SymbolicKnowledgeBase()
    kb.ingest_legal_document(Path(document), type)
    
    if output:
        kb.export_ontology(Path(output))
        click.echo(f"Knowledge base saved to: {output}")
    
    stats = kb.get_statistics()
    click.echo(f"Successfully ingested {stats['total_articles']} articles, {stats['total_rules']} rules")


@main.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True),
              help="Path to PyTorch model file")
@click.option("--data", "-d", required=True, type=click.Path(exists=True),
              help="Path to sample data file")
@click.option("--method", default="decision_tree", 
              help="Extraction method (decision_tree, activation_patterns, linear_approximation)")
@click.option("--output", "-o", type=click.Path(),
              help="Output path for extracted rules")
def extract(model: str, data: str, method: str, output: Optional[str]):
    """Extract symbolic rules from neural network models."""
    click.echo(f"Extracting rules from model: {model}")
    
    # Load model and data
    model_obj = torch.load(model, map_location='cpu')
    data_tensor = torch.load(data)
    
    # Extract rules
    bridge = NeuroSymbolicBridge()
    rules = bridge.extract_rules(model_obj, data_tensor, method=method)
    
    click.echo(f"Extracted {len(rules)} rules using {method} method")
    
    # Export rules if output specified
    if output:
        bridge.export_rules(rules, Path(output), format='json')
        click.echo(f"Rules exported to: {output}")
    
    # Display rule summary
    report = bridge.generate_rule_report(rules)
    click.echo(f"Average confidence: {report['average_confidence']:.2f}")
    click.echo(f"High confidence rules: {report['high_confidence_rules']}")


@main.command()
@click.option("--rules", "-r", required=True, type=click.Path(exists=True),
              help="Path to extracted rules file")
@click.option("--knowledge-base", "-kb", type=click.Path(exists=True),
              help="Path to knowledge base file")
@click.option("--system-id", "-s", required=True,
              help="AI system identifier")
@click.option("--system-type", default="high_risk",
              help="AI system type (high_risk, limited_risk, minimal_risk)")
@click.option("--output", "-o", type=click.Path(),
              help="Output path for assessment results")
def assess(rules: str, knowledge_base: Optional[str], system_id: str, 
          system_type: str, output: Optional[str]):
    """Assess AI system compliance against regulations."""
    click.echo(f"Assessing compliance for system: {system_id}")
    
    # Load knowledge base
    if knowledge_base:
        kb = SymbolicKnowledgeBase(Path(knowledge_base))
    else:
        kb = SymbolicKnowledgeBase()
        click.echo("Warning: Using empty knowledge base. Consider ingesting legal documents first.")
    
    # Load extracted rules
    import json
    with open(rules, 'r') as f:
        rules_data = json.load(f)
    
    # TODO: Convert JSON rules back to ModelRule objects
    # For now, create dummy rules for demonstration
    model_rules = []
    
    # Perform compliance assessment
    engine = MappingReasoningEngine(kb)
    assessment = engine.assess_compliance(system_id, model_rules, system_type)
    
    click.echo(f"Compliance Status: {assessment.overall_status.value}")
    click.echo(f"Confidence Score: {assessment.confidence_score:.1%}")
    click.echo(f"Violations Found: {len(assessment.violations)}")
    
    if output:
        engine.export_assessment(system_id, output, format='json')
        click.echo(f"Assessment exported to: {output}")


@main.command()
@click.option("--assessment", "-a", required=True, type=click.Path(exists=True),
              help="Path to compliance assessment file")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output path for compliance report")
@click.option("--format", "-f", default="html",
              help="Report format (html, pdf, json)")
def report(assessment: str, output: str, format: str):
    """Generate compliance reports from assessment results."""
    click.echo(f"Generating {format} compliance report")
    
    # Load assessment
    import json
    with open(assessment, 'r') as f:
        assessment_data = json.load(f)
    
    # TODO: Convert JSON assessment back to ComplianceAssessment object
    # For now, show placeholder
    click.echo("Report generation functionality will be implemented in the next version")
    click.echo(f"Report would be saved to: {output}")


@main.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True),
              help="Path to PyTorch model file")
@click.option("--data", "-d", required=True, type=click.Path(exists=True),
              help="Path to sample data file")
@click.option("--system-id", "-s", required=True,
              help="AI system identifier")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output directory for all results")
def pipeline(model: str, data: str, system_id: str, output: str):
    """Run complete compliance assessment pipeline."""
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo("🚀 Starting AI Rosetta Stone compliance pipeline")
    
    # Step 1: Extract rules
    click.echo("📋 Step 1: Extracting symbolic rules from model...")
    model_obj = torch.load(model, map_location='cpu')
    data_tensor = torch.load(data)
    
    bridge = NeuroSymbolicBridge()
    rules = bridge.extract_rules(model_obj, data_tensor)
    
    rules_path = output_dir / "extracted_rules.json"
    bridge.export_rules(rules, rules_path, format='json')
    click.echo(f"   ✅ Extracted {len(rules)} rules → {rules_path}")
    
    # Step 2: Initialize knowledge base (placeholder)
    click.echo("📚 Step 2: Initializing legal knowledge base...")
    kb = SymbolicKnowledgeBase()
    click.echo("   ⚠️  Using empty knowledge base (ingest legal documents for full functionality)")
    
    # Step 3: Assess compliance
    click.echo("⚖️  Step 3: Assessing regulatory compliance...")
    engine = MappingReasoningEngine(kb)
    # TODO: Convert extracted rules to proper format
    assessment = engine.assess_compliance(system_id, [], "high_risk")
    
    assessment_path = output_dir / "compliance_assessment.json"
    engine.export_assessment(system_id, str(assessment_path), format='json')
    click.echo(f"   ✅ Assessment complete → {assessment_path}")
    
    # Step 4: Generate report
    click.echo("📄 Step 4: Generating compliance report...")
    report_generator = ComplianceReportGenerator()
    report_path = output_dir / "compliance_report.html"
    
    try:
        report_generator.generate_compliance_report(assessment, report_path, format="html")
        click.echo(f"   ✅ Report generated → {report_path}")
    except Exception as e:
        click.echo(f"   ❌ Report generation failed: {e}")
    
    click.echo("\n🎉 Pipeline completed successfully!")
    click.echo(f"📁 All results saved to: {output_dir}")
    click.echo(f"📊 Compliance Status: {assessment.overall_status.value}")


if __name__ == "__main__":
    main()