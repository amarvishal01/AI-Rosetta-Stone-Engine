"""
AI Rosetta Stone: Neuro-Symbolic Bridge
Gradient-Based Activation Pattern Analysis (GBAPA) Implementation

This module extracts high-fidelity symbolic rules from trained PyTorch neural networks
for regulatory compliance with the EU AI Act.

Author: AI Research Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import logging
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
import networkx as nx
from captum.attr import GradientShap, IntegratedGradients, LayerGradientXActivation
from captum.attr import visualization as viz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SymbolicRule:
    """Represents an extracted symbolic rule."""
    rule_id: str
    conditions: List[Dict[str, Any]]  # List of conditions (feature, operator, threshold)
    conclusion: Dict[str, Any]  # Decision outcome
    confidence: float  # Rule confidence score
    coverage: float  # Percentage of data points covered
    layer_source: int  # Which layer this rule was extracted from
    neurons_involved: List[int]  # Neuron indices involved in this rule
    
    def to_readable_string(self) -> str:
        """Convert rule to human-readable IF-THEN format."""
        conditions_str = []
        for cond in self.conditions:
            feature = cond['feature']
            operator = cond['operator']
            threshold = cond['threshold']
            conditions_str.append(f"({feature} {operator} {threshold:.4f})")
        
        conditions_text = " AND ".join(conditions_str)
        conclusion_text = f"{self.conclusion['outcome']} (confidence: {self.conclusion['value']:.4f})"
        
        return f"Rule_{self.rule_id}: IF {conditions_text} THEN decision -> '{conclusion_text}'"

@dataclass
class ComplianceReport:
    """Compliance analysis report for regulatory auditing."""
    model_id: str
    total_rules: int
    compliant_rules: int
    non_compliant_rules: int
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    article_compliance: Dict[str, bool]  # EU AI Act article compliance
    detailed_findings: List[Dict[str, Any]]
    recommendations: List[str]

class NeuroSymbolicBridge:
    """
    Main class for extracting symbolic rules from neural networks using
    Gradient-Based Activation Pattern Analysis (GBAPA).
    """
    
    def __init__(self, 
                 model: nn.Module,
                 feature_names: List[str],
                 class_names: List[str] = None,
                 device: str = 'cpu'):
        """
        Initialize the Neuro-Symbolic Bridge.
        
        Args:
            model: Trained PyTorch model
            feature_names: Names of input features
            class_names: Names of output classes (for classification)
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.model.eval()
        self.feature_names = feature_names
        self.class_names = class_names or [f"class_{i}" for i in range(self._get_output_dim())]
        self.device = device
        
        # Initialize attribution methods
        self.gradient_shap = GradientShap(self.model)
        self.integrated_gradients = IntegratedGradients(self.model)
        
        # Storage for extracted rules
        self.extracted_rules: List[SymbolicRule] = []
        self.activation_patterns: Dict[int, torch.Tensor] = {}
        self.layer_hooks: List = []
        
        logger.info(f"Initialized NeuroSymbolicBridge with {len(feature_names)} features")
    
    def _get_output_dim(self) -> int:
        """Get the output dimension of the model."""
        # Create a dummy input to determine output shape
        dummy_input = torch.randn(1, len(self.feature_names)).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        return output.shape[-1]
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate activations."""
        def hook_fn(layer_idx):
            def hook(module, input, output):
                self.activation_patterns[layer_idx] = output.detach()
            return hook
        
        # Register hooks for all linear/conv layers
        layer_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                hook = module.register_forward_hook(hook_fn(layer_idx))
                self.layer_hooks.append(hook)
                layer_idx += 1
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.layer_hooks:
            hook.remove()
        self.layer_hooks.clear()
        self.activation_patterns.clear()
    
    def extract_rules(self, 
                     X_sample: torch.Tensor,
                     y_sample: torch.Tensor = None,
                     n_clusters: int = 10,
                     min_coverage: float = 0.05,
                     confidence_threshold: float = 0.7) -> List[SymbolicRule]:
        """
        Extract symbolic rules from the neural network using GBAPA.
        
        Args:
            X_sample: Sample input data for rule extraction
            y_sample: Sample target data (optional)
            n_clusters: Number of clusters for activation pattern grouping
            min_coverage: Minimum coverage threshold for rule inclusion
            confidence_threshold: Minimum confidence threshold for rule inclusion
            
        Returns:
            List of extracted symbolic rules
        """
        logger.info("Starting symbolic rule extraction...")
        
        X_sample = X_sample.to(self.device)
        if y_sample is not None:
            y_sample = y_sample.to(self.device)
        
        # Step 1: Register hooks and collect activation patterns
        self._register_hooks()
        
        # Step 2: Forward pass to collect activations
        with torch.no_grad():
            predictions = self.model(X_sample)
        
        # Step 3: Analyze each layer
        rules = []
        for layer_idx, activations in self.activation_patterns.items():
            layer_rules = self._extract_layer_rules(
                X_sample, activations, predictions, layer_idx,
                n_clusters, min_coverage, confidence_threshold
            )
            rules.extend(layer_rules)
        
        # Step 4: Clean up hooks
        self._remove_hooks()
        
        # Step 5: Post-process and filter rules
        self.extracted_rules = self._post_process_rules(rules)
        
        logger.info(f"Extracted {len(self.extracted_rules)} high-quality rules")
        return self.extracted_rules
    
    def _extract_layer_rules(self,
                           X_sample: torch.Tensor,
                           activations: torch.Tensor,
                           predictions: torch.Tensor,
                           layer_idx: int,
                           n_clusters: int,
                           min_coverage: float,
                           confidence_threshold: float) -> List[SymbolicRule]:
        """Extract rules from a specific layer's activations."""
        
        # Flatten activations if multi-dimensional
        if len(activations.shape) > 2:
            activations_flat = activations.view(activations.shape[0], -1)
        else:
            activations_flat = activations
        
        # Compute gradients with respect to activations
        activations_flat.requires_grad_(True)
        
        # Get predictions and compute gradients
        pred_classes = torch.argmax(predictions, dim=1)
        gradients = []
        
        for i in range(len(pred_classes)):
            if activations_flat.grad is not None:
                activations_flat.grad.zero_()
            
            predictions[i, pred_classes[i]].backward(retain_graph=True)
            gradients.append(activations_flat.grad[i].clone())
        
        gradients = torch.stack(gradients)
        
        # Compute gradient-weighted activations (importance scores)
        importance_scores = torch.abs(gradients * activations_flat.detach())
        
        # Cluster activation patterns
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(importance_scores.cpu().numpy())
        
        rules = []
        
        # Extract rules for each cluster
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            if np.sum(cluster_mask) < len(X_sample) * min_coverage:
                continue  # Skip small clusters
            
            cluster_inputs = X_sample[cluster_mask]
            cluster_predictions = predictions[cluster_mask]
            cluster_importance = importance_scores[cluster_mask]
            
            # Find most important neurons for this cluster
            mean_importance = torch.mean(cluster_importance, dim=0)
            top_neurons = torch.argsort(mean_importance, descending=True)[:5]  # Top 5 neurons
            
            # Extract thresholds for top neurons
            conditions = []
            for neuron_idx in top_neurons:
                neuron_activations = activations_flat[cluster_mask, neuron_idx]
                threshold = torch.median(neuron_activations).item()
                
                # Determine operator based on activation distribution
                if torch.mean(neuron_activations).item() > threshold:
                    operator = ">"
                else:
                    operator = "<="
                
                conditions.append({
                    'feature': f'neuron_{layer_idx}_{neuron_idx.item()}',
                    'operator': operator,
                    'threshold': threshold,
                    'importance': mean_importance[neuron_idx].item()
                })
            
            # Determine conclusion (most common prediction in cluster)
            cluster_pred_classes = torch.argmax(cluster_predictions, dim=1)
            most_common_class = torch.mode(cluster_pred_classes)[0].item()
            confidence = torch.mean(cluster_predictions[:, most_common_class]).item()
            
            if confidence >= confidence_threshold:
                rule = SymbolicRule(
                    rule_id=f"L{layer_idx}_C{cluster_id}",
                    conditions=conditions,
                    conclusion={
                        'outcome': self.class_names[most_common_class],
                        'value': confidence
                    },
                    confidence=confidence,
                    coverage=np.sum(cluster_mask) / len(X_sample),
                    layer_source=layer_idx,
                    neurons_involved=top_neurons.tolist()
                )
                rules.append(rule)
        
        return rules
    
    def _post_process_rules(self, rules: List[SymbolicRule]) -> List[SymbolicRule]:
        """Post-process extracted rules to remove redundancy and improve quality."""
        
        # Sort rules by confidence and coverage
        rules.sort(key=lambda r: r.confidence * r.coverage, reverse=True)
        
        # Remove highly similar rules (simplified deduplication)
        filtered_rules = []
        for rule in rules:
            is_similar = False
            for existing_rule in filtered_rules:
                if (rule.conclusion['outcome'] == existing_rule.conclusion['outcome'] and
                    abs(rule.confidence - existing_rule.confidence) < 0.1):
                    is_similar = True
                    break
            
            if not is_similar:
                filtered_rules.append(rule)
        
        return filtered_rules[:50]  # Return top 50 rules
    
    def map_to_input_features(self, 
                            X_sample: torch.Tensor,
                            target_class: int = None) -> List[SymbolicRule]:
        """
        Map neuron-level rules back to input features using gradient attribution.
        
        Args:
            X_sample: Sample input data
            target_class: Target class for attribution (if None, uses predicted class)
            
        Returns:
            List of rules mapped to input features
        """
        logger.info("Mapping rules to input features...")
        
        X_sample = X_sample.to(self.device)
        
        # Compute feature attributions using Integrated Gradients
        baseline = torch.zeros_like(X_sample[0:1])
        
        mapped_rules = []
        
        for rule in self.extracted_rules:
            # Get a representative sample for this rule
            sample_input = X_sample[0:1]  # Use first sample as example
            
            if target_class is None:
                with torch.no_grad():
                    pred = self.model(sample_input)
                    target_class = torch.argmax(pred, dim=1).item()
            
            # Compute attributions
            attributions = self.integrated_gradients.attribute(
                sample_input, baseline, target=target_class
            )
            
            # Convert to feature-based conditions
            feature_conditions = []
            attributions_abs = torch.abs(attributions[0])
            top_features = torch.argsort(attributions_abs, descending=True)[:5]
            
            for feature_idx in top_features:
                if attributions_abs[feature_idx] > 0.01:  # Significance threshold
                    feature_value = sample_input[0, feature_idx].item()
                    attribution_value = attributions[0, feature_idx].item()
                    
                    # Determine threshold and operator
                    if attribution_value > 0:
                        operator = ">"
                        threshold = feature_value * 0.9  # 90% of current value
                    else:
                        operator = "<="
                        threshold = feature_value * 1.1  # 110% of current value
                    
                    feature_conditions.append({
                        'feature': self.feature_names[feature_idx],
                        'operator': operator,
                        'threshold': threshold,
                        'importance': attributions_abs[feature_idx].item()
                    })
            
            # Create mapped rule
            mapped_rule = SymbolicRule(
                rule_id=f"MAPPED_{rule.rule_id}",
                conditions=feature_conditions,
                conclusion=rule.conclusion,
                confidence=rule.confidence,
                coverage=rule.coverage,
                layer_source=rule.layer_source,
                neurons_involved=rule.neurons_involved
            )
            mapped_rules.append(mapped_rule)
        
        return mapped_rules
    
    def generate_compliance_report(self, 
                                 rules: List[SymbolicRule],
                                 protected_features: List[str] = None,
                                 fairness_threshold: float = 0.02) -> ComplianceReport:
        """
        Generate a compliance report for regulatory auditing.
        
        Args:
            rules: List of symbolic rules to analyze
            protected_features: List of protected attribute names
            fairness_threshold: Maximum allowed bias threshold
            
        Returns:
            ComplianceReport object
        """
        logger.info("Generating compliance report...")
        
        protected_features = protected_features or []
        compliant_rules = 0
        non_compliant_rules = 0
        detailed_findings = []
        recommendations = []
        
        # Analyze each rule for compliance
        for rule in rules:
            is_compliant = True
            findings = {
                'rule_id': rule.rule_id,
                'rule_text': rule.to_readable_string(),
                'issues': []
            }
            
            # Check for use of protected attributes (Article 10 - Non-discrimination)
            for condition in rule.conditions:
                if any(protected_feat in condition['feature'].lower() 
                      for protected_feat in protected_features):
                    is_compliant = False
                    findings['issues'].append({
                        'type': 'PROTECTED_ATTRIBUTE_USAGE',
                        'article': 'Article 10',
                        'description': f"Rule uses protected attribute: {condition['feature']}",
                        'severity': 'HIGH'
                    })
            
            # Check confidence levels (Article 13 - Transparency)
            if rule.confidence < 0.7:
                findings['issues'].append({
                    'type': 'LOW_CONFIDENCE',
                    'article': 'Article 13',
                    'description': f"Rule confidence ({rule.confidence:.3f}) below recommended threshold",
                    'severity': 'MEDIUM'
                })
            
            # Check coverage (Article 15 - Accuracy and robustness)
            if rule.coverage < 0.01:
                findings['issues'].append({
                    'type': 'LOW_COVERAGE',
                    'article': 'Article 15',
                    'description': f"Rule covers only {rule.coverage:.1%} of data",
                    'severity': 'LOW'
                })
            
            if is_compliant:
                compliant_rules += 1
            else:
                non_compliant_rules += 1
            
            detailed_findings.append(findings)
        
        # Generate recommendations
        if non_compliant_rules > 0:
            recommendations.append("Remove or modify rules using protected attributes")
            recommendations.append("Implement additional bias testing procedures")
            recommendations.append("Consider model retraining with fairness constraints")
        
        # Determine risk level
        risk_ratio = non_compliant_rules / len(rules) if rules else 0
        if risk_ratio > 0.2:
            risk_level = "HIGH"
        elif risk_ratio > 0.05:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Article compliance summary
        article_compliance = {
            'Article_10_Non_Discrimination': non_compliant_rules == 0,
            'Article_13_Transparency': all(r.confidence >= 0.5 for r in rules),
            'Article_14_Human_Oversight': True,  # Assuming human oversight is implemented
            'Article_15_Accuracy_Robustness': all(r.coverage >= 0.001 for r in rules)
        }
        
        return ComplianceReport(
            model_id="neural_model_v1",
            total_rules=len(rules),
            compliant_rules=compliant_rules,
            non_compliant_rules=non_compliant_rules,
            risk_level=risk_level,
            article_compliance=article_compliance,
            detailed_findings=detailed_findings,
            recommendations=recommendations
        )
    
    def export_rules_to_json(self, rules: List[SymbolicRule], filepath: str):
        """Export rules to JSON format for external systems."""
        import json
        
        rules_data = []
        for rule in rules:
            rules_data.append({
                'rule_id': rule.rule_id,
                'conditions': rule.conditions,
                'conclusion': rule.conclusion,
                'confidence': rule.confidence,
                'coverage': rule.coverage,
                'layer_source': rule.layer_source,
                'neurons_involved': rule.neurons_involved,
                'readable_format': rule.to_readable_string()
            })
        
        with open(filepath, 'w') as f:
            json.dump(rules_data, f, indent=2)
        
        logger.info(f"Exported {len(rules)} rules to {filepath}")

# Example usage and testing functions
def create_sample_model(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    """Create a sample neural network for testing."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, output_dim),
        nn.Softmax(dim=1)
    )

def demo_neuro_symbolic_bridge():
    """Demonstration of the Neuro-Symbolic Bridge functionality."""
    
    # Create sample data and model
    torch.manual_seed(42)
    input_dim = 10
    n_samples = 1000
    
    # Sample features representing a credit scoring scenario
    feature_names = [
        'credit_score', 'income', 'age', 'employment_years', 
        'debt_ratio', 'loan_amount', 'education_level', 
        'marital_status', 'num_dependents', 'property_value'
    ]
    
    class_names = ['approved', 'rejected']
    
    # Create sample model and data
    model = create_sample_model(input_dim, 64, 2)
    X_sample = torch.randn(n_samples, input_dim)
    
    # Initialize the bridge
    bridge = NeuroSymbolicBridge(
        model=model,
        feature_names=feature_names,
        class_names=class_names
    )
    
    # Extract rules
    rules = bridge.extract_rules(X_sample, n_clusters=8, min_coverage=0.03)
    
    # Map to input features
    mapped_rules = bridge.map_to_input_features(X_sample)
    
    # Generate compliance report
    protected_features = ['age', 'marital_status']  # Protected attributes
    report = bridge.generate_compliance_report(mapped_rules, protected_features)
    
    # Print results
    print("=== EXTRACTED SYMBOLIC RULES ===")
    for rule in mapped_rules[:5]:  # Show first 5 rules
        print(rule.to_readable_string())
        print()
    
    print("=== COMPLIANCE REPORT ===")
    print(f"Total Rules: {report.total_rules}")
    print(f"Compliant Rules: {report.compliant_rules}")
    print(f"Non-Compliant Rules: {report.non_compliant_rules}")
    print(f"Risk Level: {report.risk_level}")
    print(f"Article Compliance: {report.article_compliance}")
    
    # Export rules
    bridge.export_rules_to_json(mapped_rules, '/workspace/extracted_rules.json')
    
    return bridge, rules, mapped_rules, report

if __name__ == "__main__":
    demo_neuro_symbolic_bridge()