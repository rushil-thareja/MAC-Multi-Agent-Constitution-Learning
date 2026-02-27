"""
Utilities for phrase-level processing and conversion between phrases and tokens.
"""

import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass


@dataclass
class PhraseSpan:
    """Represents a phrase span with start/end positions."""
    text: str
    start_char: int
    end_char: int
    

class PhraseTokenConverter:
    """Converts between phrase lists and token indices for compatibility."""
    
    def __init__(self):
        """Initialize converter."""
        pass
    
    def find_phrase_in_tokens(self, phrase: str, tokens: List[str], 
                             token_info: List) -> List[int]:
        """
        Find phrase in token list and return token indices.
        
        Args:
            phrase: Phrase to find
            tokens: List of tokens
            token_info: List of TokenInfo objects with positions
            
        Returns:
            List of token indices that make up this phrase
        """
        if not phrase.strip():
            return []
        
        # Try to find exact phrase match first
        phrase_lower = phrase.lower()
        
        # Look for consecutive tokens that form this phrase
        for start_idx in range(len(tokens)):
            for end_idx in range(start_idx + 1, len(tokens) + 1):
                token_sequence = " ".join(tokens[start_idx:end_idx])
                if token_sequence.lower() == phrase_lower:
                    return list(range(start_idx, end_idx))
                
                # Also try without spaces for compound phrases
                token_sequence_no_spaces = "".join(tokens[start_idx:end_idx])
                if token_sequence_no_spaces.lower() == phrase_lower.replace(" ", ""):
                    return list(range(start_idx, end_idx))
        
        # If exact match not found, try partial matches
        for start_idx in range(len(tokens)):
            for end_idx in range(start_idx + 1, min(start_idx + 10, len(tokens) + 1)):
                token_sequence = " ".join(tokens[start_idx:end_idx])
                if phrase_lower in token_sequence.lower() or token_sequence.lower() in phrase_lower:
                    return list(range(start_idx, end_idx))
        
        return []
    
    def phrases_to_token_indices(self, phrases: List[str], tokens: List[str],
                                token_info: List) -> List[int]:
        """
        Convert list of phrases to token indices.
        
        Args:
            phrases: List of phrase strings
            tokens: List of tokens
            token_info: Token info with positions
            
        Returns:
            List of unique token indices
        """
        all_indices = set()
        
        for phrase in phrases:
            indices = self.find_phrase_in_tokens(phrase, tokens, token_info)
            all_indices.update(indices)
        
        return sorted(list(all_indices))
    
    def extract_phrases_from_text(self, text: str, gold_spans: List) -> List[str]:
        """
        Extract privacy phrases from text using gold span annotations.
        
        Args:
            text: Original text
            gold_spans: List of Span objects with start/end positions
            
        Returns:
            List of privacy phrase strings
        """
        phrases = []
        
        for span in gold_spans:
            # Extract text using character positions
            if span.start < len(text) and span.end <= len(text):
                phrase_text = text[span.start:span.end].strip()
                if phrase_text:
                    phrases.append(phrase_text)
        
        # Remove duplicates while preserving order
        unique_phrases = []
        seen = set()
        for phrase in phrases:
            if phrase not in seen:
                unique_phrases.append(phrase)
                seen.add(phrase)
        
        return unique_phrases
    
    def get_phrase_context(self, phrase: str, text: str, context_chars: int = 100) -> str:
        """
        Get context around a phrase in text.
        
        Args:
            phrase: Phrase to find context for
            text: Full text
            context_chars: Number of characters before/after for context
            
        Returns:
            Context string with phrase highlighted
        """
        phrase_lower = phrase.lower()
        text_lower = text.lower()
        
        # Find phrase in text
        start_pos = text_lower.find(phrase_lower)
        if start_pos == -1:
            return f"Context not found for phrase: {phrase}"
        
        end_pos = start_pos + len(phrase)
        
        # Get context
        context_start = max(0, start_pos - context_chars)
        context_end = min(len(text), end_pos + context_chars)
        
        context = text[context_start:context_end]
        
        # Highlight the phrase
        phrase_in_context = text[start_pos:end_pos]
        highlighted_context = context.replace(phrase_in_context, f"**{phrase_in_context}**")
        
        return highlighted_context
    
    def generate_error_context(self, fn_phrases: List[str], fp_phrases: List[str],
                              text: str) -> str:
        """
        Generate context description for FN/FP phrases.

        Args:
            fn_phrases: False negative phrases
            fp_phrases: False positive phrases
            text: Original text

        Returns:
            Context description string
        """
        # Take first few phrases without filtering
        filtered_fn = fn_phrases[:5]
        filtered_fp = fp_phrases[:5]

        context_entries = []

        if filtered_fn:
            for phrase in filtered_fn:
                # Skip grouped character descriptions
                if phrase.startswith(('letters:', 'digits:', 'punctuation:')):
                    entry = f'''        Failure type: (FN)
        Phrase : "Multiple single characters: {phrase}"
        Context : "N/A"'''
                else:
                    context = self.get_phrase_context(phrase, text, 30)  # Reduced context size
                    entry = f'''        Failure type: (FN)
        Phrase : "{phrase}"
        Context : "{context}"'''
                context_entries.append(entry)

        if filtered_fp:
            for phrase in filtered_fp:
                if phrase.startswith(('letters:', 'digits:', 'punctuation:')):
                    entry = f'''        Failure type: (FP)
        Phrase : "Multiple single characters: {phrase}"
        Context : "N/A"'''
                else:
                    context = self.get_phrase_context(phrase, text, 30)  # Reduced context size
                    entry = f'''        Failure type: (FP)
        Phrase : "{phrase}"
        Context : "{context}"'''
                context_entries.append(entry)

        # Build final formatted context
        if context_entries:
            entries_text = ",\n\n".join(context_entries)
            return f"Context:\n    {{\n{entries_text}\n    }}"
        else:
            return "Context:\n    {\n    }"


def create_phrase_converter() -> PhraseTokenConverter:
    """
    Factory function to create phrase-token converter.
    
    Returns:
        PhraseTokenConverter instance
    """
    return PhraseTokenConverter()