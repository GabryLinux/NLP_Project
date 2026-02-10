import re
import spacy
from collections import defaultdict
import json

class Utilities:
    _nlp = spacy.load("en_core_web_sm")
    
    # It computes the average message length in terms of number of tokens (only alphabetic tokens are considered) for a list of messages.
    # It is based on the linguistic-based tokenizer spacy
    @staticmethod
    def avg_msg_length(documents: list[str]) -> float:
        total_length = 0

        for text in documents:
            doc = Utilities._nlp(text)
            total_length += len(list(token.lemma_.lower() for token in doc if token.is_alpha))

        avg_length = (total_length / len(documents)) if len(documents) > 0 else 0.0
        return round(avg_length, 4)
    
    # It computes the total number of tokens (only alphabetic tokens are considered) for a list of messages.
    # It is based on the linguistic-based tokenizer spacy
    @staticmethod
    def total_number_of_tokens(documents: list[str]) -> int:
        total_tokens = 0

        for text in documents:
            doc = Utilities._nlp(text)
            total_tokens += len(list(token.lemma_.lower() for token in doc if token.is_alpha))

        return total_tokens

    @staticmethod
    def extract_json(text):
        try:
            match = re.search(r'(\{.*\})', text, re.DOTALL)

            if match:
                json_str = match.group(1)
                return json.loads(json_str)
                
        except (json.JSONDecodeError, Exception) as e:
            return {"Result": "ERROR", "Error": str(e)}
        

    @staticmethod
    def safe_float(val, default=float('nan')):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default 