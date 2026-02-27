"""
Constitution management system for versioned privacy rules.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConstitutionChange:
    """Represents a change to the constitution."""
    version: int
    timestamp: str
    action: str  # 'add' or 'edit'
    rule_index: Optional[int]
    old_rule: Optional[str]
    new_rule: str
    doc_id: str
    iter_id: int
    rationale: str


class Constitution:
    """Manages versioned privacy rules."""
    
    def __init__(self, base_dir: Path, config: Dict = None):
        """
        Initialize constitution manager.
        
        Args:
            base_dir: Base directory for constitution files
            config: Configuration dictionary with initial_constitution_path
        """
        self.base_dir = Path(base_dir)
        self.constitutions_dir = self.base_dir / "constitutions"
        self.constitutions_dir.mkdir(exist_ok=True)
        
        self.version = 0
        self.rules = []
        self.change_history = []
        
        # Load initial constitution if enabled and specified in config
        if (config and 
            config.get('use_initial_constitution', False) and 
            config.get('initial_constitution_path')):
            self._load_initial_constitution(config['initial_constitution_path'])
        
        # Create/save initial constitution
        self._save_constitution()
    
    def add_rule(self, rule_text: str, doc_id: str, iter_id: int, 
                 rationale: str = "") -> int:
        """
        Add a new rule to the constitution.
        
        Args:
            rule_text: Text of the new rule
            doc_id: Document ID where rule was proposed
            iter_id: Iteration ID where rule was proposed
            rationale: Reason for adding this rule
            
        Returns:
            New constitution version number
        """
        change = ConstitutionChange(
            version=self.version + 1,
            timestamp=datetime.now().isoformat(),
            action="add",
            rule_index=None,
            old_rule=None,
            new_rule=rule_text,
            doc_id=doc_id,
            iter_id=iter_id,
            rationale=rationale
        )
        
        self.rules.append(rule_text)
        self.change_history.append(change)
        self.version += 1
        
        self._save_constitution()
        return self.version
    
    def edit_rule(self, rule_index: int, new_rule_text: str, doc_id: str,
                  iter_id: int, rationale: str = "") -> int:
        """
        Edit an existing rule.
        
        Args:
            rule_index: Index of rule to edit
            new_rule_text: New text for the rule
            doc_id: Document ID where edit was proposed
            iter_id: Iteration ID where edit was proposed
            rationale: Reason for editing this rule
            
        Returns:
            New constitution version number
        """
        if rule_index < 0 or rule_index >= len(self.rules):
            raise ValueError(f"Invalid rule index: {rule_index}")
        
        old_rule = self.rules[rule_index]
        
        change = ConstitutionChange(
            version=self.version + 1,
            timestamp=datetime.now().isoformat(),
            action="edit",
            rule_index=rule_index,
            old_rule=old_rule,
            new_rule=new_rule_text,
            doc_id=doc_id,
            iter_id=iter_id,
            rationale=rationale
        )
        
        self.rules[rule_index] = new_rule_text
        self.change_history.append(change)
        self.version += 1
        
        self._save_constitution()
        return self.version
    
    def remove_rule(self, rule_index: int, doc_id: str, iter_id: int, 
                   rationale: str = "") -> int:
        """
        Remove an existing rule.
        
        Args:
            rule_index: Index of rule to remove
            doc_id: Document ID where removal was proposed
            iter_id: Iteration ID where removal was proposed
            rationale: Reason for removing this rule
            
        Returns:
            New constitution version number
        """
        if rule_index < 0 or rule_index >= len(self.rules):
            raise ValueError(f"Invalid rule index: {rule_index}")
        
        old_rule = self.rules[rule_index]
        
        change = ConstitutionChange(
            version=self.version + 1,
            timestamp=datetime.now().isoformat(),
            action="remove",
            rule_index=rule_index,
            old_rule=old_rule,
            new_rule=None,
            doc_id=doc_id,
            iter_id=iter_id,
            rationale=rationale
        )
        
        # Remove the rule from the list
        self.rules.pop(rule_index)
        self.change_history.append(change)
        self.version += 1
        
        self._save_constitution()
        return self.version
    
    def apply_change(self, change_proposal: Dict, doc_id: str, iter_id: int) -> int:
        """
        Apply a change proposal to the constitution.
        
        Args:
            change_proposal: Dict with action, rule_index, rule_text
            doc_id: Document ID
            iter_id: Iteration ID
            
        Returns:
            New constitution version number
        """
        action = change_proposal.get('action')
        rule_text = change_proposal.get('rule_text')
        
        if action == 'add':
            return self.add_rule(rule_text, doc_id, iter_id, 
                               f"Added to reduce FN/FP in {doc_id}")
        elif action == 'edit':
            rule_index = change_proposal.get('rule_index')
            if rule_index is None:
                raise ValueError("Edit action requires rule_index")
            return self.edit_rule(rule_index, rule_text, doc_id, iter_id,
                                f"Edited to improve performance in {doc_id}")
        elif action == 'remove':
            rule_index = change_proposal.get('rule_index')
            if rule_index is None:
                raise ValueError("Remove action requires rule_index")
            return self.remove_rule(rule_index, doc_id, iter_id,
                                  f"Removed to reduce FP/improve performance in {doc_id}")
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def get_text(self) -> str:
        """
        Get constitution as formatted text.
        
        Returns:
            Constitution text with rules
        """
        if not self.rules:
            return f"# Constitution v{self.version:04d} (empty)\n# No specific rules defined yet. Use general privacy principles to identify private information."
        
        lines = [
            f"# Constitution v{self.version:04d}",
            f"# created: {datetime.now().isoformat()}",
            f"# rules: {len(self.rules)}",
            ""
        ]
        
        for i, rule in enumerate(self.rules):
            lines.append(f"{i+1}. {rule}")
        
        return "\n".join(lines)
    
    def get_indexed_text(self) -> str:
        """
        Get constitution with rule indices for editing prompts.
        
        Returns:
            Constitution text with explicit indices
        """
        if not self.rules:
            return "Constitution is empty."
        
        lines = []
        for i, rule in enumerate(self.rules):
            lines.append(f"[{i}] {rule}")
        
        return "\n".join(lines)
    
    def load_constitution(self, version: Optional[int] = None) -> None:
        """
        Load constitution from disk.
        
        Args:
            version: Specific version to load, or latest if None
        """
        if version is None:
            # Find latest version
            constitution_files = list(self.constitutions_dir.glob("c_v*.txt"))
            if not constitution_files:
                return  # No constitutions exist yet
            
            versions = []
            for f in constitution_files:
                try:
                    v = int(f.stem.split('_v')[1])
                    versions.append(v)
                except ValueError:
                    continue
            
            version = max(versions) if versions else 0
        
        constitution_file = self.constitutions_dir / f"c_v{version:04d}.txt"
        metadata_file = self.constitutions_dir / f"c_v{version:04d}.json"
        
        if constitution_file.exists():
            with open(constitution_file, 'r') as f:
                content = f.read()
            
            # Parse rules from content
            self.rules = []
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '. ' in line:
                    # Extract rule text after number
                    rule_text = line.split('. ', 1)[1]
                    self.rules.append(rule_text)
            
            self.version = version
        
        # Load metadata if exists
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.change_history = [
                    ConstitutionChange(**change) for change in metadata.get('changes', [])
                ]
    
    def _save_constitution(self) -> None:
        """Save current constitution to disk."""
        # Save text version
        constitution_file = self.constitutions_dir / f"c_v{self.version:04d}.txt"
        with open(constitution_file, 'w') as f:
            f.write(self.get_text())
        
        # Save metadata
        metadata_file = self.constitutions_dir / f"c_v{self.version:04d}.json"
        metadata = {
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'rules_count': len(self.rules),
            'changes': [
                {
                    'version': change.version,
                    'timestamp': change.timestamp,
                    'action': change.action,
                    'rule_index': change.rule_index,
                    'old_rule': change.old_rule,
                    'new_rule': change.new_rule,
                    'doc_id': change.doc_id,
                    'iter_id': change.iter_id,
                    'rationale': change.rationale
                }
                for change in self.change_history
            ]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_initial_constitution(self, initial_constitution_path: str) -> None:
        """
        Load rules from initial constitution file.
        
        Args:
            initial_constitution_path: Path to initial constitution file
        """
        constitution_path = Path(initial_constitution_path)
        
        if not constitution_path.exists():
            print(f"Warning: Initial constitution file not found at {constitution_path}")
            return
        
        with open(constitution_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse rules from content, ignoring comments and empty lines
        rules = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Check if line starts with "Rule N:" or contains rule-like content
                if line.lower().startswith('rule ') and ':' in line:
                    # Extract rule text after "Rule N: "
                    rule_text = line.split(':', 1)[1].strip()
                    if rule_text:
                        rules.append(rule_text)
                elif not line.startswith('# ') and len(line) > 20:  # Likely a rule without "Rule N:" prefix
                    # Remove any existing numbering (e.g., "1. Mark..." -> "Mark...")
                    cleaned_rule = line
                    if line[0].isdigit() and '. ' in line:
                        cleaned_rule = line.split('. ', 1)[1]
                    rules.append(cleaned_rule)
        
        if rules:
            self.rules = rules
            self.version = 1  # Start at version 1 instead of 0 when loading initial constitution
            
            # Create single change history entry for initial constitution load
            change = ConstitutionChange(
                version=1,
                timestamp=datetime.now().isoformat(),
                action="add",
                rule_index=None,
                old_rule=None,
                new_rule=f"Loaded {len(rules)} rules from initial constitution",
                doc_id="initial",
                iter_id=0,
                rationale="Loaded from initial constitution file"
            )
            self.change_history.append(change)
            
            print(f"Loaded {len(rules)} rules from initial constitution: {initial_constitution_path}")
        else:
            print(f"No rules found in initial constitution file: {initial_constitution_path}")
    
    def get_diff(self, old_version: int, new_version: int) -> str:
        """
        Get diff between two constitution versions.
        
        Args:
            old_version: Old version number
            new_version: New version number
            
        Returns:
            Human-readable diff
        """
        # Find changes between versions
        relevant_changes = [
            change for change in self.change_history
            if old_version < change.version <= new_version
        ]
        
        if not relevant_changes:
            return "No changes between versions."
        
        diff_lines = [f"Changes from v{old_version:04d} to v{new_version:04d}:"]
        
        for change in relevant_changes:
            if change.action == 'add':
                diff_lines.append(f"+ Added rule: {change.new_rule}")
            elif change.action == 'edit':
                diff_lines.append(f"~ Edited rule {change.rule_index}:")
                diff_lines.append(f"  - {change.old_rule}")
                diff_lines.append(f"  + {change.new_rule}")
        
        return "\n".join(diff_lines)
    
    def get_recent_changes(self, count: int = 5) -> List[ConstitutionChange]:
        """
        Get most recent changes.
        
        Args:
            count: Number of recent changes to return
            
        Returns:
            List of recent ConstitutionChange objects
        """
        return self.change_history[-count:] if self.change_history else []


def create_constitution_manager(run_dir: Path, config: Dict = None) -> Constitution:
    """
    Factory function to create constitution manager.
    
    Args:
        run_dir: Run directory path
        config: Configuration dictionary
        
    Returns:
        Constitution instance
    """
    return Constitution(run_dir, config)