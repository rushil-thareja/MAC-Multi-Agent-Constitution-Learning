"""
Batch processing utilities for epoch-based constitutional training.
Handles processing multiple documents together and aggregating results.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from .phrase_utils import create_phrase_converter

logger = logging.getLogger(__name__)


@dataclass
class BatchDocument:
    """Represents a document in a training batch."""
    doc_id: str
    text: str
    gold_phrases: List[str]
    txt_path: Optional[Path] = None
    ann_path: Optional[Path] = None
    # New fields for example-based (DSPy-style) mode
    example: Any = None        # Original Example object (if from examples mode)
    gold_output: Any = None    # Generic gold output (str, List, Dict, etc.)


@dataclass
class BatchResult:
    """Results from processing a batch of documents."""
    batch_id: str
    epoch: int
    batch_index: int
    doc_ids: List[str]
    aggregated_fn_phrases: List[str]
    aggregated_fp_phrases: List[str]
    batch_context: str
    batch_metrics: Dict[str, float]
    individual_results: List[Dict[str, Any]]
    # New fields for example-based (DSPy-style) mode
    error_reports: Optional[List[Any]] = None    # List[ErrorReport] when in examples mode
    batch_score: float = 0.0                     # Mean metric score across examples


class BatchAnnotationProcessor:
    """Processes batches of documents for annotation and evaluation."""

    def __init__(self, agents: Dict[str, Any], config: Dict[str, Any] = None):
        """
        Initialize batch annotation processor.

        Args:
            agents: Dictionary of agent instances
            config: Configuration dictionary (optional, for caching)
        """
        self.agents = agents
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def _aggregate_batch_results(self, individual_results: List[Dict[str, Any]], 
                                epoch: int, batch_index: int, context: str) -> BatchResult:
        """
        Aggregate results from individual documents into batch-level metrics.
        
        Args:
            individual_results: List of individual document results
            epoch: Current epoch number
            batch_index: Current batch index
            context: Processing context (e.g., "no_constitution" or "constitution_v5")
            
        Returns:
            BatchResult with aggregated data
        """
        batch_id = f"epoch_{epoch:02d}_batch_{batch_index:02d}"
        self.logger.info(f"📊 AGGREGATING BATCH RESULTS: {batch_id}")
        
        # Aggregate FN and FP phrases
        all_fn_phrases = []
        all_fp_phrases = []
        individual_counts = []
        
        for i, result in enumerate(individual_results):
            evaluation = result['evaluation']
            doc_fn = evaluation.get('fn_phrases', [])
            doc_fp = evaluation.get('fp_phrases', [])
            
            all_fn_phrases.extend(doc_fn)
            all_fp_phrases.extend(doc_fp)
            
            individual_counts.append({
                'doc_id': result['doc_id'],
                'fn': len(doc_fn),
                'fp': len(doc_fp),
                'tp': evaluation['tp']
            })
            
            self.logger.info(f"   📄 DOC {result['doc_id']}: FN={len(doc_fn)}, FP={len(doc_fp)}, TP={evaluation['tp']}")
        
        # Remove duplicates while preserving original phrases
        aggregated_fn = list(dict.fromkeys(all_fn_phrases))  # Preserves order, removes duplicates
        aggregated_fp = list(dict.fromkeys(all_fp_phrases))
        
        self.logger.info(f"📈 AGGREGATION RESULTS:")
        self.logger.info(f"   📊 Raw FN phrases: {len(all_fn_phrases)} total")
        self.logger.info(f"   📊 Raw FP phrases: {len(all_fp_phrases)} total")  
        self.logger.info(f"   🎯 Unique FN phrases: {len(aggregated_fn)}")
        self.logger.info(f"   🎯 Unique FP phrases: {len(aggregated_fp)}")
        self.logger.info(f"   📊 FN deduplication: {len(all_fn_phrases)} → {len(aggregated_fn)} ({len(all_fn_phrases)-len(aggregated_fn)} removed)")
        self.logger.info(f"   📊 FP deduplication: {len(all_fp_phrases)} → {len(aggregated_fp)} ({len(all_fp_phrases)-len(aggregated_fp)} removed)")
        
        # Calculate batch-level metrics
        total_tp = sum(result['evaluation']['tp'] for result in individual_results)
        total_fp = sum(result['evaluation']['fp'] for result in individual_results) 
        total_fn = sum(result['evaluation']['fn'] for result in individual_results)
        
        batch_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        batch_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        batch_f1 = 2 * batch_precision * batch_recall / (batch_precision + batch_recall) if (batch_precision + batch_recall) > 0 else 0.0
        
        batch_metrics = {
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'precision': batch_precision,
            'recall': batch_recall,
            'f1': batch_f1,
            'num_docs': len(individual_results)
        }
        
        # Generate batch context for rule generation
        batch_context = self._generate_batch_context(aggregated_fn, aggregated_fp, individual_results)
        
        doc_ids = [result['doc_id'] for result in individual_results]
        
        self.logger.info(f"✅ BATCH {batch_id} AGGREGATION COMPLETE:")
        self.logger.info(f"   🎯 Final Metrics: F1={batch_f1:.3f}, Precision={batch_precision:.3f}, Recall={batch_recall:.3f}")
        self.logger.info(f"   📊 Error Counts: TP={total_tp}, FP={total_fp}, FN={total_fn}")
        self.logger.info(f"   📋 Unique Error Phrases: FN={len(aggregated_fn)}, FP={len(aggregated_fp)}")
        self.logger.info(f"   📄 Documents: {doc_ids}")
        
        # Log sample error phrases for debugging
        if aggregated_fn:
            self.logger.info(f"   📝 Sample FN phrases: {aggregated_fn[:5]}{'...' if len(aggregated_fn) > 5 else ''}")
        if aggregated_fp:
            self.logger.info(f"   📝 Sample FP phrases: {aggregated_fp[:5]}{'...' if len(aggregated_fp) > 5 else ''}")
        
        return BatchResult(
            batch_id=batch_id,
            epoch=epoch,
            batch_index=batch_index,
            doc_ids=doc_ids,
            aggregated_fn_phrases=aggregated_fn,
            aggregated_fp_phrases=aggregated_fp,
            batch_context=batch_context,
            batch_metrics=batch_metrics,
            individual_results=individual_results
        )
    
    def _generate_batch_context(self, fn_phrases: List[str], fp_phrases: List[str], 
                               individual_results: List[Dict[str, Any]]) -> str:
        """
        Generate context description for batch-level FN/FP phrases.
        
        Args:
            fn_phrases: Aggregated false negative phrases
            fp_phrases: Aggregated false positive phrases
            individual_results: Individual document results
            
        Returns:
            Context description string for rule generation
        """
        phrase_converter = create_phrase_converter()

        context_lines = []
        
        if fn_phrases:
            context_lines.append(f"Batch False Negatives ({len(fn_phrases)} unique phrases):")
            # Show examples with document context
            for i, phrase in enumerate(fn_phrases[:5]):  # Limit to first 5
                # Find a document that contains this phrase for context
                doc_with_phrase = None
                for result in individual_results:
                    if phrase in result['evaluation'].get('fn_phrases', []):
                        doc_with_phrase = result
                        break
                
                if doc_with_phrase:
                    context = phrase_converter.get_phrase_context(phrase, doc_with_phrase['text'], 30)
                    context_lines.append(f"  - {context} (from {doc_with_phrase['doc_id']})")
                else:
                    context_lines.append(f"  - {phrase}")
        
        if fp_phrases:
            context_lines.append(f"Batch False Positives ({len(fp_phrases)} unique phrases):")
            # Show examples with document context
            for i, phrase in enumerate(fp_phrases[:5]):  # Limit to first 5
                # Find a document that contains this phrase for context
                doc_with_phrase = None
                for result in individual_results:
                    if phrase in result['evaluation'].get('fp_phrases', []):
                        doc_with_phrase = result
                        break
                
                if doc_with_phrase:
                    context = phrase_converter.get_phrase_context(phrase, doc_with_phrase['text'], 30)
                    context_lines.append(f"  - {context} (from {doc_with_phrase['doc_id']})")
                else:
                    context_lines.append(f"  - {phrase}")
        
        # Add batch-level statistics
        context_lines.append(f"\nBatch Statistics:")
        context_lines.append(f"  - Documents processed: {len(individual_results)}")
        context_lines.append(f"  - Total FN phrases: {len(fn_phrases)}")
        context_lines.append(f"  - Total FP phrases: {len(fp_phrases)}")
        
        return "\n".join(context_lines)

    def _format_failure_context(self, limited_fn_phrases: List[str], limited_fp_phrases: List[str],
                               individual_results: List[Dict[str, Any]]) -> str:
        """
        Format limited FN/FP phrases as clear failure cases with document context.

        Args:
            limited_fn_phrases: Limited false negative phrases
            limited_fp_phrases: Limited false positive phrases
            individual_results: Individual document results

        Returns:
            Formatted context string with failure cases and contexts
        """
        from .phrase_utils import create_phrase_converter
        phrase_converter = create_phrase_converter()

        context_entries = []

        # Format FN phrases
        for phrase in limited_fn_phrases:
            # Find a document that contains this phrase for context
            doc_with_phrase = None
            for result in individual_results:
                if phrase in result['evaluation'].get('fn_phrases', []):
                    doc_with_phrase = result
                    break

            if doc_with_phrase:
                doc_context = phrase_converter.get_phrase_context(phrase, doc_with_phrase['text'], 80)
                entry = f'''        Failure type: (FN)
        Phrase : "{phrase}"
        Context : "{doc_context}"'''
                context_entries.append(entry)

        # Format FP phrases
        for phrase in limited_fp_phrases:
            # Find a document that contains this phrase for context
            doc_with_phrase = None
            for result in individual_results:
                if phrase in result['evaluation'].get('fp_phrases', []):
                    doc_with_phrase = result
                    break

            if doc_with_phrase:
                doc_context = phrase_converter.get_phrase_context(phrase, doc_with_phrase['text'], 80)
                entry = f'''        Failure type: (FP)
        Phrase : "{phrase}"
        Context : "{doc_context}"'''
                context_entries.append(entry)

        # Build final formatted context
        if context_entries:
            entries_text = ",\n\n".join(context_entries)
            return f"Context:\n    {{\n{entries_text}\n    }}"
        else:
            return "Context:\n    {\n    }"

    # ==================================================================
    # Example-based (DSPy-style) methods
    # ==================================================================

    def prepare_batch_from_examples(self, examples) -> List[BatchDocument]:
        """Create BatchDocuments from Example objects.

        If Example.output is List[str], also fills gold_phrases for
        compatibility with phrase-based error aggregation.
        """
        batch_docs = []
        for ex in examples:
            gold_phrases = ex.output if isinstance(ex.output, list) else []
            batch_docs.append(BatchDocument(
                doc_id=ex.id,
                text=ex.input,
                gold_phrases=gold_phrases,
                example=ex,
                gold_output=ex.output,
            ))
        return batch_docs

    def process_batch_with_metric(self, batch_docs: List[BatchDocument],
                                  constitution, epoch: int, batch_index: int,
                                  metric, error_analyzer) -> BatchResult:
        """Process a batch using the user-supplied metric + error analyzer.

        1. Annotate all examples (concurrently when async is available).
        2. Run metric(prediction, gold) -> score.
        3. Run error_analyzer.analyze() -> ErrorReport.
        4. Aggregate into BatchResult (including error_reports + batch_score).
        """
        from .example import ErrorReport  # type hint only

        constitution_text = constitution.get_text() if constitution and hasattr(constitution, 'get_text') else None
        has_rules = constitution_text and len(getattr(constitution, 'rules', [])) > 0

        # ── Batch annotation: concurrent when async client is available ──
        annotator = self.agents.get('annotator')
        use_concurrent = (
            annotator and
            hasattr(annotator, 'async_client') and
            annotator.async_client is not None and
            len(batch_docs) > 1
        )

        if use_concurrent:
            results = self._annotate_batch_concurrent(
                annotator, batch_docs, constitution_text if has_rules else None
            )
        else:
            results = self._annotate_batch_sequential(
                annotator, batch_docs, constitution_text if has_rules else None
            )

        individual_results = []
        error_reports = []

        for doc, (prediction, reasoning) in zip(batch_docs, results):

            # Unwrap single-element list for scalar gold outputs
            if isinstance(prediction, list) and not isinstance(doc.gold_output, list):
                if len(prediction) == 1:
                    prediction = prediction[0]
                elif len(prediction) == 0:
                    prediction = ""

            # Score
            score = metric(prediction, doc.gold_output)

            # Error analysis
            report = error_analyzer.analyze(
                prediction, doc.gold_output,
                input_text=doc.text,
                example_id=doc.doc_id
            )
            report.score = score
            report.reasoning = reasoning
            error_reports.append(report)

            # Build individual result (compat with existing pipeline)
            fn_phrases = [e.description for e in report.errors if e.error_type == "MISSED"]
            fp_phrases = [e.description for e in report.errors if e.error_type in ("SPURIOUS", "WRONG")]
            individual_results.append({
                'doc_id': doc.doc_id,
                'predicted_phrases': prediction if isinstance(prediction, list) else [str(prediction)],
                'gold_phrases': doc.gold_phrases,
                'evaluation': {
                    'tp': 0, 'fp': len(fp_phrases), 'fn': len(fn_phrases),
                    'precision': score, 'recall': score, 'f1': score,
                    'fn_phrases': fn_phrases, 'fp_phrases': fp_phrases,
                },
                'text': doc.text,
            })

        # Aggregate
        batch_score = sum(r.score for r in error_reports) / len(error_reports) if error_reports else 0.0

        # Build context from error reports
        batch_context = self._generate_error_report_context(error_reports, batch_score)

        # Collect fn/fp for compat
        all_fn = []
        all_fp = []
        for r in individual_results:
            all_fn.extend(r['evaluation'].get('fn_phrases', []))
            all_fp.extend(r['evaluation'].get('fp_phrases', []))

        batch_metrics = {
            'tp': 0, 'fp': len(all_fp), 'fn': len(all_fn),
            'precision': batch_score, 'recall': batch_score,
            'f1': batch_score, 'num_docs': len(individual_results),
        }

        return BatchResult(
            batch_id=f"epoch_{epoch:02d}_batch_{batch_index:02d}",
            epoch=epoch,
            batch_index=batch_index,
            doc_ids=[doc.doc_id for doc in batch_docs],
            aggregated_fn_phrases=list(dict.fromkeys(all_fn)),
            aggregated_fp_phrases=list(dict.fromkeys(all_fp)),
            batch_context=batch_context,
            batch_metrics=batch_metrics,
            individual_results=individual_results,
            error_reports=error_reports,
            batch_score=batch_score,
        )

    def _annotate_batch_sequential(self, annotator, batch_docs, constitution_text):
        """Annotate docs one by one, returning list of (prediction, reasoning)."""
        results = []
        for doc in batch_docs:
            if hasattr(annotator, 'process_with_reasoning'):
                prediction, reasoning = annotator.process_with_reasoning(doc.text, constitution_text)
            else:
                prediction = annotator.process(doc.text, constitution_text)
                reasoning = ''
            results.append((prediction, reasoning))
        return results

    def _annotate_batch_concurrent(self, annotator, batch_docs, constitution_text):
        """Annotate docs concurrently via asyncio, returning list of (prediction, reasoning).

        Falls back to sequential on any error.
        """
        import asyncio

        async def _process_one(doc):
            const_block = (
                f"**CONSTITUTION RULES (follow these to guide your response):**\n{constitution_text}"
                if constitution_text
                else "If no rules are provided, use your best judgement."
            )
            system_prompt, user_prompt = annotator.prompt_template.format(
                CONSTITUTION_BLOCK=const_block,
                TEXT_CONTENT=doc.text
            )
            response = await annotator._call_llm_async(system_prompt, user_prompt)
            result = annotator._extract_json(response)
            output = annotator._extract_output(result)
            reasoning = result.get('reasoning', '') if isinstance(result, dict) else ''
            return output, reasoning

        async def _run_all():
            sem = asyncio.Semaphore(min(len(batch_docs), 8))
            async def _with_sem(doc):
                async with sem:
                    try:
                        return await _process_one(doc)
                    except RuntimeError:
                        raise  # Config errors (e.g. bad temperature) — stop immediately
                    except Exception as e:
                        logger.warning(f"Concurrent annotation failed for {doc.doc_id}: {e}")
                        return [], ''

            results = await asyncio.gather(*[_with_sem(d) for d in batch_docs],
                                           return_exceptions=True)
            # If any result is a fatal config error, propagate it
            for r in results:
                if isinstance(r, RuntimeError):
                    raise r
                if isinstance(r, BaseException):
                    raise r

            # Check for total batch failure (all returned empty)
            all_empty = all(
                (isinstance(r, tuple) and r[0] in ([], '', None))
                for r in results
            )
            if all_empty and len(batch_docs) > 0:
                raise RuntimeError(
                    f"All {len(batch_docs)} examples in batch returned empty predictions. "
                    f"This usually means the model is rejecting every request. "
                    f"Check model compatibility and API parameters."
                )

            return [r if isinstance(r, tuple) else ([], '') for r in results]

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _run_all())
                    return list(future.result())
            else:
                return list(asyncio.run(_run_all()))
        except Exception as e:
            logger.warning(f"Concurrent batch annotation failed: {e}, falling back to sequential")
            return self._annotate_batch_sequential(annotator, batch_docs, constitution_text)

    def _generate_error_report_context(self, error_reports, batch_score: float) -> str:
        """Format ErrorReport list into a context string for the rule proposer/editor.

        Shows full examples (input, prediction, gold, errors) sorted by score
        ascending (worst first) so the proposer can write targeted rules.
        Respects config['error_context'] knobs and an optional custom formatter.
        """
        # Read config knobs
        ec_cfg = getattr(self, 'config', {}).get('error_context', {})
        show_full = ec_cfg.get('show_full_examples', True)
        max_examples = ec_cfg.get('max_error_examples', 8)
        max_input_chars = ec_cfg.get('max_input_chars', 1500)

        # Custom formatter path
        formatter = getattr(self, 'error_context_formatter', None)

        if not show_full and formatter is None:
            # Legacy compact format
            lines = [f"Batch score: {batch_score:.2f}/1.0 across {len(error_reports)} examples"]
            total_errors = sum(len(r.errors) for r in error_reports)
            lines.append(f"{total_errors} total errors:\n")
            error_num = 0
            for report in error_reports:
                for err in report.errors:
                    error_num += 1
                    if error_num > 20:
                        lines.append(f"  ... and {total_errors - 20} more errors")
                        break
                    ctx = f"\n    Context: {err.context}" if err.context else ""
                    lines.append(f"  Error {error_num} ({err.error_type}):")
                    lines.append(f"    {err.description}{ctx}")
                if error_num > 20:
                    break
            return "\n".join(lines)

        # Sort by score ascending (worst first)
        sorted_reports = sorted(error_reports, key=lambda r: r.score)
        shown = sorted_reports[:max_examples]

        lines = [
            f"Batch score: {batch_score:.2f}/1.0 across {len(error_reports)} examples "
            f"({len(shown)} worst shown)"
        ]

        for i, report in enumerate(shown, 1):
            if formatter is not None:
                lines.append(f"\n--- Example {i} (score: {report.score:.2f}) ---")
                lines.append(formatter(report))
                continue

            # Default rich format
            input_text = report.input_text or ""
            if len(input_text) > max_input_chars:
                input_text = input_text[:max_input_chars] + "..."

            lines.append(f"\n--- Example {i} (score: {report.score:.2f}) ---")
            lines.append(f"Input: \"{input_text}\"")
            # Include the model's reasoning so the rule proposer can see WHERE it went wrong
            reasoning = getattr(report, 'reasoning', '') or ''
            if reasoning:
                max_reasoning = ec_cfg.get('max_reasoning_chars', 1000)
                if len(reasoning) > max_reasoning:
                    reasoning = reasoning[:max_reasoning] + "..."
                lines.append(f"Model reasoning: {reasoning}")
            lines.append(f"Predicted: {report.prediction}")
            lines.append(f"Gold: {report.gold}")
            if report.errors:
                lines.append("Errors:")
                for err in report.errors:
                    lines.append(f"  - ({err.error_type}) {err.description}")

        return "\n".join(lines)


class BatchDecisionMaker:
    """Makes constitutional decisions based on batch-level error patterns."""
    
    def __init__(self, agents: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize batch decision maker.
        
        Args:
            agents: Dictionary of agent instances
            config: Configuration dictionary
        """
        self.agents = agents
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Rule modification constraints from config
        self.enable_rule_editing = config['algorithm'].get('enable_rule_editing', True)
        self.enable_rule_removal = config['algorithm'].get('enable_rule_removal', True)
        self.min_rule_age_for_edit = config['algorithm'].get('min_rule_age_for_edit', 2)
        self.max_consecutive_removes = config['algorithm'].get('max_consecutive_removes', 2)
        self.max_total_rules = config['algorithm'].get('max_total_rules', None)
        
        # Track consecutive removals
        self.consecutive_removals = 0
        
        # Log configuration validation
        self.logger.info("🔧 BATCH DECISION MAKER CONFIGURATION:")
        self.logger.info(f"   📊 enable_rule_editing: {config['algorithm'].get('enable_rule_editing', 'NOT SET')}")
        self.logger.info(f"   📊 enable_rule_removal: {config['algorithm'].get('enable_rule_removal', 'NOT SET')}")
        self.logger.info(f"   📊 max_total_rules: {self.max_total_rules or 'unlimited'}")
        self.logger.info(f"   📊 error_limits configured: {config.get('error_limits', 'NOT SET')}")
        self.logger.info(f"   📊 Available agents: {list(agents.keys())}")

        # Validate that required agents exist for unified decision system
        required_unified_agents = ['new_decision_agent', 'new_rule_proposer', 'new_rule_editor']
        missing_agents = [agent for agent in required_unified_agents if agent not in agents]
        if missing_agents:
            self.logger.warning(f"⚠️  MISSING UNIFIED AGENTS: {missing_agents}")
        else:
            self.logger.info("✅ All required unified decision agents available")
    
    def _limit_error_phrases(self, fn_phrases: List[str], fp_phrases: List[str]) -> Tuple[List[str], List[str]]:
        """
        Limit FN and FP phrases based on configuration limits.
        
        Args:
            fn_phrases: List of false negative phrases
            fp_phrases: List of false positive phrases
            
        Returns:
            Tuple of (limited_fn_phrases, limited_fp_phrases)
        """
        # Get limits from config
        error_limits = self.config.get('error_limits', {})
        max_fn = error_limits.get('max_fn_phrases', 15)
        max_fp = error_limits.get('max_fp_phrases', 15)
        selection_method = error_limits.get('phrase_selection', 'random')
        
        # Apply limits to FN phrases
        limited_fn = fn_phrases.copy()
        if len(limited_fn) > max_fn:
            if selection_method == 'random':
                limited_fn = random.sample(limited_fn, max_fn)
            elif selection_method == 'first':
                limited_fn = limited_fn[:max_fn]
            else:  # 'frequent' - for now, treat same as first
                limited_fn = limited_fn[:max_fn]
        
        # Apply limits to FP phrases  
        limited_fp = fp_phrases.copy()
        if len(limited_fp) > max_fp:
            if selection_method == 'random':
                limited_fp = random.sample(limited_fp, max_fp)
            elif selection_method == 'first':
                limited_fp = limited_fp[:max_fp]
            else:  # 'frequent' - for now, treat same as first
                limited_fp = limited_fp[:max_fp]
        
        if len(fn_phrases) > max_fn or len(fp_phrases) > max_fp:
            self.logger.info(f"🔢 LIMITED PHRASES: FN {len(fn_phrases)}→{len(limited_fn)}, "
                           f"FP {len(fp_phrases)}→{len(limited_fp)} (method: {selection_method})")
        
        return limited_fn, limited_fp
    
    def _format_failure_context(self, limited_fn_phrases: List[str], limited_fp_phrases: List[str],
                               individual_results: List[Dict[str, Any]]) -> str:
        """
        Format limited FN/FP phrases as clear failure cases with document context.

        Args:
            limited_fn_phrases: Limited false negative phrases
            limited_fp_phrases: Limited false positive phrases
            individual_results: Individual document results

        Returns:
            Formatted context string with failure cases and contexts
        """
        from .phrase_utils import create_phrase_converter
        phrase_converter = create_phrase_converter()

        context_entries = []

        # Format FN phrases
        for phrase in limited_fn_phrases:
            # Find a document that contains this phrase for context
            doc_with_phrase = None
            for result in individual_results:
                if phrase in result['evaluation'].get('fn_phrases', []):
                    doc_with_phrase = result
                    break

            if doc_with_phrase:
                doc_context = phrase_converter.get_phrase_context(phrase, doc_with_phrase['text'], 80)
                entry = f'''        Failure type: (FN)
        Phrase : "{phrase}"
        Context : "{doc_context}"'''
                context_entries.append(entry)

        # Format FP phrases
        for phrase in limited_fp_phrases:
            # Find a document that contains this phrase for context
            doc_with_phrase = None
            for result in individual_results:
                if phrase in result['evaluation'].get('fp_phrases', []):
                    doc_with_phrase = result
                    break

            if doc_with_phrase:
                doc_context = phrase_converter.get_phrase_context(phrase, doc_with_phrase['text'], 80)
                entry = f'''        Failure type: (FP)
        Phrase : "{phrase}"
        Context : "{doc_context}"'''
                context_entries.append(entry)

        # Build final formatted context
        if context_entries:
            entries_text = ",\n\n".join(context_entries)
            return f"Context:\n    {{\n{entries_text}\n    }}"
        else:
            return "Context:\n    {\n    }"

    def make_constitutional_decision(self, constitution: Any, batch_result: BatchResult,
                                   previous_rejections: List[Dict] = None) -> Dict[str, Any]:
        """
        Make decision on how to modify constitution based on batch results.
        
        Args:
            constitution: Current constitution object
            batch_result: BatchResult with aggregated FN/FP data
            previous_rejections: List of previous rejected rule attempts
            
        Returns:
            Decision dict with action, rule_index, and reasoning
        """
        self.logger.info(f"🔍 MAKING CONSTITUTIONAL DECISION for {batch_result.batch_id}")
        self.logger.info(f"📊 Constitution status: {len(constitution.rules)} rules")
        self.logger.info(f"📊 Error counts: FN={len(batch_result.aggregated_fn_phrases)}, FP={len(batch_result.aggregated_fp_phrases)}")
        
        # Show sample phrases for context
        if batch_result.aggregated_fn_phrases:
            self.logger.info(f"📝 Sample FN phrases: {batch_result.aggregated_fn_phrases[:3]}...")
        if batch_result.aggregated_fp_phrases:
            self.logger.info(f"📝 Sample FP phrases: {batch_result.aggregated_fp_phrases[:3]}...")
        
        # If no errors, no action needed
        if not batch_result.aggregated_fn_phrases and not batch_result.aggregated_fp_phrases:
            self.logger.info("✅ No FN/FP errors found, no constitutional change needed")
            return {
                'action': 'no_change',
                'rule_index': None,
                'reasoning': 'Perfect performance achieved - no errors to address'
            }
        
        # If constitution is empty, must add first rule (FIRST-TIME LOGIC)
        if len(constitution.rules) == 0:
            self.logger.info("🚀 EMPTY CONSTITUTION - must add first rule (bypassing bifurcated system for safety)")
            return {
                'action': 'add',
                'rule_index': None,
                'reasoning': 'Constitution is empty - adding first rule to address error patterns'
            }
        
        # NEW UNIFIED DECISION SYSTEM
        self.logger.info("🎯 USING NEW UNIFIED DECISION SYSTEM")

        try:
            self.logger.info("🔢 STEP 1: Limiting FN/FP phrases based on configuration")
            # Step 1: Limit FN/FP phrases based on configuration
            limited_fn_phrases, limited_fp_phrases = self._limit_error_phrases(
                batch_result.aggregated_fn_phrases,
                batch_result.aggregated_fp_phrases
            )
            self.logger.info(f"📉 LIMITED PHRASES: FN {len(batch_result.aggregated_fn_phrases)}→{len(limited_fn_phrases)}, FP {len(batch_result.aggregated_fp_phrases)}→{len(limited_fp_phrases)}")

            # Store limited phrases for decision result
            self._current_limited_fn_phrases = limited_fn_phrases
            self._current_limited_fp_phrases = limited_fp_phrases

            # Get counts for statistics
            fn_count = len(batch_result.aggregated_fn_phrases)
            fp_count = len(batch_result.aggregated_fp_phrases)

            self.logger.info("🎯 STEP 2: Calling NewDecisionAgent (unified decision)")
            self.logger.info(f"📊 Input: fn_count={fn_count}, fp_count={fp_count}, total_rules={len(constitution.rules)}")
            self.logger.info(f"📝 Limited phrases: FN={limited_fn_phrases[:3]}..., FP={limited_fp_phrases[:3]}...")

            # Single unified decision call - returns action AND rule_index in one call
            decision = self.agents['new_decision_agent'].decide(
                constitution_text=constitution.get_indexed_text(),
                fn_count=fn_count,
                fp_count=fp_count,
                fn_phrases=limited_fn_phrases,
                fp_phrases=limited_fp_phrases,
                total_rules=len(constitution.rules)
            )

            self.logger.info(f"✅ UNIFIED DECISION: action={decision.get('action')}, rule_index={decision.get('rule_index')}")
            self.logger.info(f"💭 REASONING: {decision.get('reasoning', 'N/A')}")

            # Validate decision against constraints
            self.logger.info("🔍 STEP 3: Validating decision against constraints")
            validated_decision = self._validate_decision(decision, constitution, batch_result)

            self.logger.info(f"✅ FINAL DECISION for {batch_result.batch_id}: {validated_decision['action']} "
                           f"(rule_index: {validated_decision.get('rule_index', 'N/A')})")
            self.logger.info(f"💭 FINAL REASONING: {validated_decision['reasoning']}")

            # Update consecutive removal tracking
            if validated_decision['action'] == 'remove':
                self.consecutive_removals += 1
            else:
                self.consecutive_removals = 0

            self.logger.info("🎉 UNIFIED DECISION SYSTEM COMPLETED SUCCESSFULLY")
            return validated_decision

        except Exception as e:
            self.logger.error(f"💥 DECISION SYSTEM EXCEPTION: {str(e)}")
            self.logger.error(f"📍 Exception type: {type(e).__name__}")
            import traceback
            self.logger.error(f"📍 Full traceback: {traceback.format_exc()}")

            # Smart fallback based on rule count
            total_rules = len(constitution.rules) if constitution and hasattr(constitution, 'rules') else 0
            if total_rules >= self.max_total_rules:
                self.logger.error("📍 Rule limit reached - falling back to safe EDIT action")
                return {
                    'action': 'edit',
                    'rule_index': total_rules - 1,
                    'reasoning': f'Decision agent failed ({e}), at rule limit - defaulting to EDIT last rule'
                }
            else:
                self.logger.error("📍 Falling back to NO CHANGE action")
                return {
                    'action': 'no_change',
                    'rule_index': None,
                    'reasoning': f'Decision agent failed ({e}), defaulting to no constitutional change'
                }
    
    def _validate_decision(self, decision: Dict[str, Any], constitution: Any, 
                          batch_result: BatchResult) -> Dict[str, Any]:
        """
        Validate and potentially modify decision based on configuration constraints.
        
        Args:
            decision: Original decision from DecisionAgent
            constitution: Current constitution
            batch_result: Batch results
            
        Returns:
            Validated decision dict
        """
        action = decision.get('action', 'add')
        rule_index = decision.get('rule_index')
        reasoning = decision.get('reasoning', 'No reasoning provided')
        
        # Check rule limit first - this is the most important constraint
        total_rules = len(constitution.rules) if constitution and hasattr(constitution, 'rules') else 0
        at_rule_limit = total_rules >= self.max_total_rules
        
        # If at rule limit, only EDIT and REMOVE are allowed
        if at_rule_limit and action == 'add':
            self.logger.warning(f"🚫 Rule limit reached ({total_rules}/{self.max_total_rules}), cannot ADD - changing to EDIT last rule")
            return {
                'action': 'edit',
                'rule_index': total_rules - 1,  # Edit the last rule
                'reasoning': f'Rule limit reached - editing last rule instead: {reasoning}'
            }
        
        # Validate EDIT action constraints
        if action == 'edit':
            if not self.enable_rule_editing:
                if at_rule_limit:
                    self.logger.warning("Rule editing disabled but at rule limit - no change possible")
                    return {
                        'action': 'no_change',
                        'rule_index': None,
                        'reasoning': 'Rule editing disabled and at rule limit - no constitutional change possible'
                    }
                else:
                    self.logger.warning("Rule editing disabled by config, changing to ADD")
                    return {
                        'action': 'add',
                        'rule_index': None,
                        'reasoning': 'Rule editing disabled - adding new rule instead'
                    }
            
            # Check rule age (placeholder - would need rule creation tracking)
            # For now, allow all edits
            
            # Validate rule index
            if rule_index is None or rule_index < 0 or rule_index >= len(constitution.rules):
                if at_rule_limit:
                    # At rule limit - try to edit the last valid rule instead
                    self.logger.warning(f"Invalid rule index {rule_index} for edit, at rule limit - editing last rule")
                    return {
                        'action': 'edit',
                        'rule_index': total_rules - 1,
                        'reasoning': f'Invalid edit rule index {rule_index}, at rule limit - editing last rule instead'
                    }
                else:
                    self.logger.warning(f"Invalid rule index {rule_index} for edit, changing to ADD")
                    return {
                        'action': 'add',
                        'rule_index': None,
                        'reasoning': f'Invalid edit rule index {rule_index} - adding new rule instead'
                    }
        
        # Validate REMOVE action constraints
        elif action == 'remove':
            if not self.enable_rule_removal:
                if at_rule_limit:
                    self.logger.warning("Rule removal disabled but at rule limit - no change possible")
                    return {
                        'action': 'no_change',
                        'rule_index': None,
                        'reasoning': 'Rule removal disabled and at rule limit - no constitutional change possible'
                    }
                else:
                    self.logger.warning("Rule removal disabled by config, changing to ADD")
                    return {
                        'action': 'add',
                        'rule_index': None,
                        'reasoning': 'Rule removal disabled - adding new rule instead'
                    }
            
            # Check consecutive removal limit
            if self.consecutive_removals >= self.max_consecutive_removes:
                if at_rule_limit:
                    self.logger.warning(f"Max consecutive removals reached but at rule limit - changing to EDIT")
                    return {
                        'action': 'edit',
                        'rule_index': total_rules - 1,
                        'reasoning': f'Max consecutive removals reached, at rule limit - editing last rule instead'
                    }
                else:
                    self.logger.warning(f"Max consecutive removals ({self.max_consecutive_removes}) reached, changing to ADD")
                    return {
                        'action': 'add',
                        'rule_index': None,
                        'reasoning': f'Max consecutive removals reached - adding new rule instead'
                    }
            
            # Validate rule index
            if rule_index is None or rule_index < 0 or rule_index >= len(constitution.rules):
                if at_rule_limit:
                    self.logger.warning(f"Invalid rule index {rule_index} for removal, at rule limit - no change")
                    return {
                        'action': 'no_change',
                        'rule_index': None,
                        'reasoning': f'Invalid removal rule index {rule_index}, at rule limit - no constitutional change'
                    }
                else:
                    self.logger.warning(f"Invalid rule index {rule_index} for removal, changing to ADD")
                    return {
                        'action': 'add',
                        'rule_index': None,
                        'reasoning': f'Invalid removal rule index {rule_index} - adding new rule instead'
                    }
        
        # Return validated decision with limited phrases
        decision_result = {
            'action': action,
            'rule_index': rule_index,
            'reasoning': reasoning
        }
        
        # Include limited phrases if they exist
        if hasattr(self, '_current_limited_fn_phrases'):
            decision_result['limited_fn_phrases'] = self._current_limited_fn_phrases
        if hasattr(self, '_current_limited_fp_phrases'):
            decision_result['limited_fp_phrases'] = self._current_limited_fp_phrases
            
        return decision_result


class BatchRuleModifier:
    """Executes constitutional rule modifications (ADD/EDIT/REMOVE) based on batch decisions."""
    
    def __init__(self, agents: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize batch rule modifier.
        
        Args:
            agents: Dictionary of agent instances
            config: Configuration dictionary
        """
        self.agents = agents
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def _limit_phrases(self, phrases: List[str], max_phrases: int, selection_method: str = 'random') -> List[str]:
        """
        Limit the number of phrases based on configuration.
        
        Args:
            phrases: List of phrases to limit
            max_phrases: Maximum number of phrases to return
            selection_method: Method to select phrases ('random', 'first', 'frequent')
            
        Returns:
            Limited list of phrases
        """
        if len(phrases) <= max_phrases:
            return phrases
        
        if selection_method == 'first':
            return phrases[:max_phrases]
        elif selection_method == 'frequent':
            # Count phrase frequencies and select most frequent
            from collections import Counter
            phrase_counts = Counter(phrases)
            most_frequent = [phrase for phrase, _ in phrase_counts.most_common(max_phrases)]
            return most_frequent
        else:  # default to random
            import random
            return random.sample(phrases, max_phrases)
    
    def _format_failure_context(self, limited_fn_phrases: List[str], limited_fp_phrases: List[str],
                               individual_results: List[Dict[str, Any]]) -> str:
        """
        Format limited FN/FP phrases as clear failure cases with document context.

        Args:
            limited_fn_phrases: Limited false negative phrases
            limited_fp_phrases: Limited false positive phrases
            individual_results: Individual document results

        Returns:
            Formatted context string with failure cases and contexts
        """
        from .phrase_utils import create_phrase_converter
        phrase_converter = create_phrase_converter()

        context_entries = []

        # Format FN phrases
        for phrase in limited_fn_phrases:
            # Find a document that contains this phrase for context
            doc_with_phrase = None
            for result in individual_results:
                if phrase in result['evaluation'].get('fn_phrases', []):
                    doc_with_phrase = result
                    break

            if doc_with_phrase:
                doc_context = phrase_converter.get_phrase_context(phrase, doc_with_phrase['text'], 80)
                entry = f'''        Failure type: (FN)
        Phrase : "{phrase}"
        Context : "{doc_context}"'''
                context_entries.append(entry)

        # Format FP phrases
        for phrase in limited_fp_phrases:
            # Find a document that contains this phrase for context
            doc_with_phrase = None
            for result in individual_results:
                if phrase in result['evaluation'].get('fp_phrases', []):
                    doc_with_phrase = result
                    break

            if doc_with_phrase:
                doc_context = phrase_converter.get_phrase_context(phrase, doc_with_phrase['text'], 80)
                entry = f'''        Failure type: (FP)
        Phrase : "{phrase}"
        Context : "{doc_context}"'''
                context_entries.append(entry)

        # Build final formatted context
        if context_entries:
            entries_text = ",\n\n".join(context_entries)
            return f"Context:\n    {{\n{entries_text}\n    }}"
        else:
            return "Context:\n    {\n    }"

    def execute_rule_change(self, decision: Dict[str, Any], constitution: Any, 
                           batch_result: BatchResult, previous_rejections: List[Dict] = None) -> Optional[int]:
        """
        Execute constitutional rule change based on decision.
        
        Args:
            decision: Decision dict from BatchDecisionMaker
            constitution: Current constitution object
            batch_result: BatchResult with error data
            previous_rejections: List of previous rejected attempts
            
        Returns:
            New constitution version number or None if no change made
        """
        action = decision['action']
        
        if action == 'no_change':
            self.logger.info(f"No constitutional change for {batch_result.batch_id}")
            return None
        
        self.logger.info(f"Executing {action} action for {batch_result.batch_id}")
        
        # Apply error limits configuration for decision making
        error_limits = self.config.get('error_limits', {})
        max_fn = error_limits.get('max_fn_phrases', 15)
        max_fp = error_limits.get('max_fp_phrases', 15)
        selection_method = error_limits.get('phrase_selection', 'random')

        # Limit phrases based on configuration
        limited_fn_phrases = self._limit_phrases(batch_result.aggregated_fn_phrases, max_fn, selection_method)
        limited_fp_phrases = self._limit_phrases(batch_result.aggregated_fp_phrases, max_fp, selection_method)

        # Use enriched context (full examples) when available
        if batch_result.batch_context and 'Example' in batch_result.batch_context:
            formatted_context = batch_result.batch_context
        else:
            formatted_context = self._format_failure_context(limited_fn_phrases, limited_fp_phrases, batch_result.individual_results)
        
        # Save context to runs folder for verification
        import os
        from pathlib import Path
        
        # Try to get run ID from current_run file or directory structure
        try:
            current_run_file = "runs/current_run"
            if os.path.isfile(current_run_file):
                with open(current_run_file, 'r') as f:
                    run_id = f.read().strip()
            else:
                # Fallback: get most recent run directory
                runs_dir = Path("runs")
                run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("202")]
                run_id = max(run_dirs, key=lambda x: x.name).name if run_dirs else "default"
        except Exception as e:
            self.logger.warning(f"Could not determine run_id: {e}, using 'default'")
            run_id = "default"
        
        # Create context directory and save file
        context_dir = Path(f"runs/{run_id}/context")
        context_dir.mkdir(exist_ok=True, parents=True)
        context_file = context_dir / f"{batch_result.batch_id}_context.txt"
        
        with open(context_file, 'w') as f:
            f.write(f"Batch: {batch_result.batch_id}\n")
            f.write(f"FN Phrases: {len(limited_fn_phrases)}, FP Phrases: {len(limited_fp_phrases)}\n")
            f.write(f"Action: {action}\n\n")
            f.write(formatted_context)
        
        self.logger.info(f"Saved context to {context_file}")
        
        try:
            if action == 'add':
                combined_reasoning = decision.get('reasoning', '')
                return self._execute_add_rule(constitution, batch_result, previous_rejections, combined_reasoning, 
                                            limited_fn_phrases, limited_fp_phrases, formatted_context)
            
            elif action == 'edit':
                rule_index = decision['rule_index']
                combined_reasoning = decision.get('reasoning', '')
                return self._execute_edit_rule(rule_index, constitution, batch_result, previous_rejections, combined_reasoning,
                                             limited_fn_phrases, limited_fp_phrases, formatted_context)
            
            elif action == 'remove':
                rule_index = decision['rule_index']
                return self._execute_remove_rule(rule_index, constitution, batch_result)
            
            else:
                self.logger.error(f"Unknown action: {action}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing {action} rule change: {e}")
            return None
    
    def _execute_add_rule(self, constitution: Any, batch_result: BatchResult,
                         previous_rejections: List[Dict] = None, combined_reasoning: str = "",
                         limited_fn_phrases: List[str] = None, limited_fp_phrases: List[str] = None,
                         formatted_context: str = None, temperature: float = None) -> Optional[int]:
        """Execute ADD rule action using NewRuleProposerAgent."""
        try:
            # Use limited phrases or fall back to original phrases
            fn_phrases = limited_fn_phrases if limited_fn_phrases is not None else batch_result.aggregated_fn_phrases
            fp_phrases = limited_fp_phrases if limited_fp_phrases is not None else batch_result.aggregated_fp_phrases

            # Use NewRuleProposerAgent to create new rule
            rule_proposal = self.agents['new_rule_proposer'].propose_rule(
                constitution_text=constitution.get_text(),
                error_context=formatted_context or batch_result.batch_context,
                previous_reasoning=combined_reasoning,
                previous_rejections=previous_rejections or [],
                temperature=temperature
            )
            
            rule_text = rule_proposal.get('rule_text', '').strip()
            
            if not rule_text:
                self.logger.warning("Empty rule text from agent, skipping ADD")
                return None
            
            # Basic validation (rule length)
            max_rule_length = self.config['validation'].get('max_rule_length', 300)
            if len(rule_text) > max_rule_length:
                self.logger.warning(f"Rule text too long ({len(rule_text)} > {max_rule_length}), skipping ADD")
                return None
            
            # Add rule to constitution
            new_version = constitution.add_rule(
                rule_text=rule_text,
                doc_id=batch_result.batch_id,
                iter_id=0,  # Batch-level, no iteration
                rationale=f"Added from batch {batch_result.batch_id} to address {len(batch_result.aggregated_fn_phrases)} FN and {len(batch_result.aggregated_fp_phrases)} FP errors"
            )
            
            self.logger.info(f"Added rule: '{rule_text}' -> Constitution v{new_version}")
            return new_version
            
        except Exception as e:
            self.logger.error(f"Error adding rule: {e}")
            return None
    
    def _execute_edit_rule(self, rule_index: int, constitution: Any, batch_result: BatchResult,
                          previous_rejections: List[Dict] = None, combined_reasoning: str = "",
                          limited_fn_phrases: List[str] = None, limited_fp_phrases: List[str] = None,
                          formatted_context: str = None, temperature: float = None) -> Optional[int]:
        """Execute EDIT rule action using NewRuleEditorAgent."""
        try:
            if rule_index < 0 or rule_index >= len(constitution.rules):
                self.logger.error(f"Invalid rule index for edit: {rule_index}")
                return None

            old_rule = constitution.rules[rule_index]

            # Use limited phrases or fall back to original phrases
            fn_phrases = limited_fn_phrases if limited_fn_phrases is not None else batch_result.aggregated_fn_phrases
            fp_phrases = limited_fp_phrases if limited_fp_phrases is not None else batch_result.aggregated_fp_phrases

            # Use NewRuleEditorAgent to edit existing rule
            rule_proposal = self.agents['new_rule_editor'].edit_rule(
                rule_index=rule_index,
                constitution_text=constitution.get_text(),
                error_context=formatted_context or batch_result.batch_context,
                previous_reasoning=combined_reasoning,
                previous_rejections=previous_rejections or [],
                temperature=temperature
            )
            
            new_rule_text = rule_proposal.get('rule_text', '').strip()
            
            if not new_rule_text:
                self.logger.warning("Empty rule text from agent, skipping EDIT")
                return None
            
            if new_rule_text == old_rule:
                self.logger.info("Rule text unchanged, skipping EDIT")
                return None
            
            # Basic validation (rule length)
            max_rule_length = self.config['validation'].get('max_rule_length', 300)
            if len(new_rule_text) > max_rule_length:
                self.logger.warning(f"New rule text too long ({len(new_rule_text)} > {max_rule_length}), skipping EDIT")
                return None
            
            # Edit rule in constitution
            new_version = constitution.edit_rule(
                rule_index=rule_index,
                new_rule_text=new_rule_text,
                doc_id=batch_result.batch_id,
                iter_id=0,  # Batch-level, no iteration
                rationale=f"Edited from batch {batch_result.batch_id} to address {len(batch_result.aggregated_fn_phrases)} FN and {len(batch_result.aggregated_fp_phrases)} FP errors"
            )
            
            self.logger.info(f"Edited rule {rule_index}: '{old_rule}' -> '{new_rule_text}' (Constitution v{new_version})")
            return new_version
            
        except Exception as e:
            self.logger.error(f"Error editing rule: {e}")
            return None
    
    def _execute_remove_rule(self, rule_index: int, constitution: Any, 
                           batch_result: BatchResult) -> Optional[int]:
        """Execute REMOVE rule action."""
        try:
            if rule_index < 0 or rule_index >= len(constitution.rules):
                self.logger.error(f"Invalid rule index for removal: {rule_index}")
                return None
            
            removed_rule = constitution.rules[rule_index]
            
            # Remove rule from constitution
            new_version = constitution.remove_rule(
                rule_index=rule_index,
                doc_id=batch_result.batch_id,
                iter_id=0,  # Batch-level, no iteration
                rationale=f"Removed from batch {batch_result.batch_id} due to poor performance (excessive FP generation)"
            )
            
            self.logger.info(f"Removed rule {rule_index}: '{removed_rule}' (Constitution v{new_version})")
            return new_version
            
        except Exception as e:
            self.logger.error(f"Error removing rule: {e}")
            return None


def create_batch_processor(agents: Dict[str, Any], config: Dict[str, Any] = None) -> BatchAnnotationProcessor:
    """
    Factory function to create batch annotation processor.

    Args:
        agents: Dictionary of agent instances
        config: Configuration dictionary (optional, for caching)

    Returns:
        BatchAnnotationProcessor instance
    """
    return BatchAnnotationProcessor(agents, config)


def create_batch_decision_maker(agents: Dict[str, Any], config: Dict[str, Any]) -> BatchDecisionMaker:
    """
    Factory function to create batch decision maker.
    
    Args:
        agents: Dictionary of agent instances
        config: Configuration dictionary
        
    Returns:
        BatchDecisionMaker instance
    """
    return BatchDecisionMaker(agents, config)


def create_batch_rule_modifier(agents: Dict[str, Any], config: Dict[str, Any]) -> BatchRuleModifier:
    """
    Factory function to create batch rule modifier.
    
    Args:
        agents: Dictionary of agent instances
        config: Configuration dictionary
        
    Returns:
        BatchRuleModifier instance
    """
    return BatchRuleModifier(agents, config)
