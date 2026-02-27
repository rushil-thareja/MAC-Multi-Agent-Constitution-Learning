#!/usr/bin/env python3
"""
One-to-one phrase matching evaluator using maximum bipartite matching.

Implements the containment-based matching rule where a gold phrase matches
a predicted phrase if the gold phrase is contained within the predicted phrase
(after canonicalization). Uses Hopcroft-Karp algorithm to ensure one-to-one
matching and prevent double counting.
"""

import unicodedata
import re
from typing import List, Tuple, Dict, Any, Optional, Set
from collections import deque
from dataclasses import dataclass


@dataclass
class MatchingResult:
    """Results from phrase matching evaluation."""
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    matched_pairs: List[Tuple[str, str]]  # (predicted, gold) pairs
    unmatched_predicted: List[str]
    unmatched_gold: List[str]


class PhraseMatchingEvaluator:
    """
    Evaluator for phrase extraction using one-to-one bipartite matching.
    
    Uses containment-based matching where gold ⊆ predicted counts as a match,
    with proper text canonicalization and maximum bipartite matching to prevent
    double counting.
    """
    
    def __init__(self):
        self.NIL = -1  # Sentinel value for unmatched nodes
    
    def canonicalize(self, text: str) -> str:
        """
        Canonicalize text for matching by:
        1. Unicode normalization (NFKC)
        2. Lowercase conversion
        3. Dash unification (– and — to -)
        4. Whitespace collapse (trim + collapse runs to single space)
        """
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Lowercase
        text = text.lower()
        
        # Unify dashes
        text = text.replace('–', '-').replace('—', '-')
        
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def matches(self, predicted_phrase: str, gold_phrase: str) -> bool:
        """
        Check if gold phrase matches predicted phrase using containment rule.
        Gold phrase is considered a match if it's contained in predicted phrase
        after canonicalization.
        """
        pred_canon = self.canonicalize(predicted_phrase)
        gold_canon = self.canonicalize(gold_phrase)
        
        # Empty strings don't match anything
        if not pred_canon or not gold_canon:
            return False
        
        return gold_canon in pred_canon
    
    def build_bipartite_graph(self, predicted: List[str], gold: List[str]) -> List[List[int]]:
        """
        Build bipartite graph adjacency list.
        
        Args:
            predicted: List of predicted phrases (left side of bipartite graph)
            gold: List of gold phrases (right side of bipartite graph)
            
        Returns:
            Adjacency list where adj[u] contains indices of gold phrases
            that match predicted phrase u
        """
        m = len(predicted)
        adj = [[] for _ in range(m)]
        
        for u in range(m):
            for v in range(len(gold)):
                if self.matches(predicted[u], gold[v]):
                    adj[u].append(v)
        
        return adj
    
    def hopcroft_karp(self, adj: List[List[int]], m: int, n: int) -> Tuple[List[int], List[int], int]:
        """
        Hopcroft-Karp algorithm for maximum bipartite matching.
        
        Args:
            adj: Adjacency list from left (predicted) to right (gold) nodes
            m: Number of predicted phrases (left nodes)
            n: Number of gold phrases (right nodes)
            
        Returns:
            Tuple of (pair_u, pair_v, matching_size) where:
            - pair_u[u] = gold index matched to predicted u (or NIL)
            - pair_v[v] = predicted index matched to gold v (or NIL)
            - matching_size = size of maximum matching
        """
        # Initialize pairing arrays
        pair_u = [self.NIL] * m  # Which gold is matched to predicted u
        pair_v = [self.NIL] * n  # Which predicted is matched to gold v
        dist = [0] * m           # Distance array for BFS
        
        def bfs() -> bool:
            """BFS to find augmenting paths."""
            queue = deque()
            
            # Initialize distance array and queue with free nodes
            for u in range(m):
                if pair_u[u] == self.NIL:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            
            found_augmenting = False
            
            while queue:
                u = queue.popleft()
                
                for v in adj[u]:
                    u2 = pair_v[v]  # Predicted currently matched to gold v
                    
                    if u2 == self.NIL:
                        # Found free node on right side
                        found_augmenting = True
                    elif dist[u2] == float('inf'):
                        # Found matched node, add to queue for next layer
                        dist[u2] = dist[u] + 1
                        queue.append(u2)
            
            return found_augmenting
        
        def dfs(u: int) -> bool:
            """DFS to find and augment along shortest augmenting path."""
            for v in adj[u]:
                u2 = pair_v[v]
                
                if u2 == self.NIL or (dist[u2] == dist[u] + 1 and dfs(u2)):
                    # Found augmenting path or can augment through u2
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            
            # Mark as unreachable in this iteration
            dist[u] = float('inf')
            return False
        
        matching_size = 0
        
        # Main algorithm loop
        while bfs():
            # Try to find augmenting paths starting from free nodes
            for u in range(m):
                if pair_u[u] == self.NIL and dfs(u):
                    matching_size += 1
        
        return pair_u, pair_v, matching_size
    
    def compute_metrics(self, predicted: List[str], gold: List[str]) -> MatchingResult:
        """
        Compute evaluation metrics using one-to-one bipartite matching.
        
        Args:
            predicted: List of predicted phrases
            gold: List of gold standard phrases
            
        Returns:
            MatchingResult with precision, recall, F1, and detailed breakdown
        """
        # Handle empty cases
        if not predicted and not gold:
            return MatchingResult(
                tp=0, fp=0, fn=0, tn=0,
                precision=1.0, recall=1.0, f1=1.0,
                matched_pairs=[], unmatched_predicted=[], unmatched_gold=[]
            )
        
        if not predicted:
            return MatchingResult(
                tp=0, fp=0, fn=len(gold), tn=0,
                precision=0.0, recall=0.0, f1=0.0,
                matched_pairs=[], unmatched_predicted=[], unmatched_gold=gold[:]
            )
        
        if not gold:
            return MatchingResult(
                tp=0, fp=len(predicted), fn=0, tn=0,
                precision=0.0, recall=0.0, f1=0.0,
                matched_pairs=[], unmatched_predicted=predicted[:], unmatched_gold=[]
            )
        
        m = len(predicted)
        n = len(gold)
        
        # Build bipartite graph
        adj = self.build_bipartite_graph(predicted, gold)
        
        # Find maximum matching
        pair_u, pair_v, tp = self.hopcroft_karp(adj, m, n)
        
        # Build result details
        matched_pairs = []
        unmatched_predicted = []
        unmatched_gold = []
        
        # Collect matched pairs and unmatched predicted
        for u in range(m):
            if pair_u[u] != self.NIL:
                matched_pairs.append((predicted[u], gold[pair_u[u]]))
            else:
                unmatched_predicted.append(predicted[u])
        
        # Collect unmatched gold
        for v in range(n):
            if pair_v[v] == self.NIL:
                unmatched_gold.append(gold[v])
        
        # Compute metrics
        fp = m - tp  # Predicted phrases with no matched gold
        fn = n - tp  # Gold phrases with no matched prediction
        tn = 0       # Undefined in open-set extraction; set to 0 by convention
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return MatchingResult(
            tp=tp, fp=fp, fn=fn, tn=tn,
            precision=precision, recall=recall, f1=f1,
            matched_pairs=matched_pairs,
            unmatched_predicted=unmatched_predicted,
            unmatched_gold=unmatched_gold
        )
    
    def evaluate_document(self, predicted: List[str], gold: List[str]) -> Dict[str, Any]:
        """
        Evaluate a single document and return detailed results.
        
        Args:
            predicted: List of predicted phrases for the document
            gold: List of gold standard phrases for the document
            
        Returns:
            Dictionary with evaluation results and detailed breakdown
        """
        result = self.compute_metrics(predicted, gold)
        
        return {
            'metrics': {
                'tp': result.tp,
                'fp': result.fp,
                'fn': result.fn,
                'tn': result.tn,
                'precision': result.precision,
                'recall': result.recall,
                'f1': result.f1
            },
            'details': {
                'num_predicted': len(predicted),
                'num_gold': len(gold),
                'matched_pairs': result.matched_pairs,
                'unmatched_predicted': result.unmatched_predicted,
                'unmatched_gold': result.unmatched_gold,
                'fp_phrases': result.unmatched_predicted,  # For compatibility
                'fn_phrases': result.unmatched_gold       # For compatibility
            }
        }
    
    def evaluate_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a batch of documents and compute aggregate metrics.
        
        Args:
            documents: List of documents, each with 'predicted_phrases' and 'gold_phrases'
            
        Returns:
            Dictionary with batch-level metrics and individual document results
        """
        individual_results = []
        total_tp = total_fp = total_fn = 0
        
        for doc in documents:
            predicted = doc.get('predicted_phrases', [])
            gold = doc.get('gold_phrases', [])
            doc_id = doc.get('doc_id', 'unknown')
            
            doc_result = self.evaluate_document(predicted, gold)
            doc_result['doc_id'] = doc_id
            
            individual_results.append(doc_result)
            
            # Accumulate totals
            total_tp += doc_result['metrics']['tp']
            total_fp += doc_result['metrics']['fp']
            total_fn += doc_result['metrics']['fn']
        
        # Compute aggregate metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0.0
        
        return {
            'batch_metrics': {
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1
            },
            'individual_results': individual_results,
            'num_documents': len(documents)
        }
