"""
TextProcessor.py - Text Processing Pipeline
Xử lý text theo các bước trong slides:
1. Tokenizing
2. Stopping (remove stopwords)
3. Stemming (optional)
4. N-grams (optional)
"""

import re
import unicodedata
from typing import List, Set, Dict
from underthesea import word_tokenize, sent_tokenize

class TextProcessor:
    """Text processor cho tiếng Việt"""

    def __init__(self, 
                 use_stopwords: bool = True,
                 library: str = 'underthesea'):
        self.use_stopwords = use_stopwords
        self.library = library
        self.stopwords = self._load_stopwords() if use_stopwords else set()

        # Statistics cho visualization
        self.stats = {
            'original_tokens': 0,
            'after_normalization': 0,
            'after_stopping': 0,
            'unique_tokens': set()
        }

    def _load_stopwords(self) -> Set[str]:
        """Load Vietnamese stopwords"""
        stopwords = {
            'và', 'của', 'có', 'cho', 'với', 'được', 'từ', 'trong',
            'là', 'một', 'các', 'để', 'theo', 'này', 'đó', 'những',
            'nhưng', 'hoặc', 'nếu', 'thì', 'khi', 'vì', 'do', 'bởi',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was'
        }
        return stopwords

    def normalize_text(self, text: str) -> str:
        """
        Bước 1: Normalize text
        - Chuẩn hóa Unicode (NFC)
        - Lowercase
        - Remove special characters
        - Remove extra whitespace
        """
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)

        # Lowercase
        text = text.lower()

        # Remove special characters (giữ chữ cái tiếng Việt)
        text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Bước 2: Tokenization
        - Tách text thành các từ (word tokens)
        """
        # Normalize first
        text = self.normalize_text(text)

        # Tokenize using underthesea
        if self.library == 'underthesea':
            tokens = word_tokenize(text, format="text").split()
        else:
            tokens = text.split()

        self.stats['after_normalization'] = len(tokens)

        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Bước 3: Stopping
        - Loại bỏ stopwords
        - Loại bỏ tokens quá ngắn
        """
        if not self.use_stopwords:
            return tokens

        # Remove stopwords and short tokens
        filtered_tokens = [
            t for t in tokens 
            if t not in self.stopwords and len(t) >= 2
        ]

        self.stats['after_stopping'] = len(filtered_tokens)

        return filtered_tokens

    def process(self, text: str) -> List[str]:
        """
        Main processing pipeline:
        1. Tokenize
        2. Remove stopwords
        3. Return processed tokens
        """
        # Reset stats
        self.stats['original_tokens'] = len(text.split())

        # Step 1: Tokenize
        tokens = self.tokenize(text)

        # Step 2: Remove stopwords
        tokens = self.remove_stopwords(tokens)

        # Update stats
        self.stats['unique_tokens'] = set(tokens)

        return tokens

    def process_documents(self, documents: List[str]) -> List[List[str]]:
        """Process multiple documents"""
        return [self.process(doc) for doc in documents]

    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'original_tokens': self.stats['original_tokens'],
            'after_normalization': self.stats['after_normalization'],
            'after_stopping': self.stats['after_stopping'],
            'unique_tokens': len(self.stats['unique_tokens']),
            'stopwords_removed': self.stats['after_normalization'] - self.stats['after_stopping']
        }
