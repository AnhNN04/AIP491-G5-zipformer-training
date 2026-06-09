import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DialectRouter:
    """
    Usecase/Service to classify the dialect of Vietnamese transcriptions
    based on vocabulary heuristics and return classification probabilities.
    Supported dialects: NORTHERN, CENTRAL, SOUTHERN.
    """
    
    # Southern regional words
    SOUTHERN_WORDS = {
        "hổng", "mắc cười", "nhậu", "lẹ", "bông", "vô", "nè", "nghen", "hén", "má", "ba", "xài",
        "kiếm", "kêu", "dạ", "được", "bây giờ", "mắc", "muỗng", "ly", "thơm"
    }
    
    # Central regional words
    CENTRAL_WORDS = {
        "chi", "mô", "tê", "răng", "ni", "nớ", "tui", "họ", "mệ", "ả", "ngái", "đơm", "trôông",
        "bọ", "mạ"
    }
    
    # Northern regional words
    NORTHERN_WORDS = {
        "này", "thế", "nhỉ", "nhé", "vâng", "bố", "mẹ", "quả", "bát", "thế à", "chứ", "ngon"
    }

    def classify_dialect(self, text: str) -> Dict[str, Any]:
        """
        Classifies the Vietnamese dialect of the text based on lexical indicators.
        Returns a dictionary containing the inferred dialect and probability.
        """
        if not text:
            return {
                "inferred": "NORTHERN",
                "probability": 0.70
            }

        # Normalize text: lowercase and strip punctuation
        normalized = re.sub(r"[^\w\s]", "", text.lower())
        words = normalized.split()

        southern_count = sum(1 for w in words if w in self.SOUTHERN_WORDS)
        central_count = sum(1 for w in words if w in self.CENTRAL_WORDS)
        northern_count = sum(1 for w in words if w in self.NORTHERN_WORDS)

        total_regional_hits = southern_count + central_count + northern_count

        if total_regional_hits == 0:
            # Fallback default dialect
            return {
                "inferred": "NORTHERN",
                "probability": 0.70
            }

        # Calculate probabilities based on hits relative to total regional hits
        counts = {
            "SOUTHERN": southern_count,
            "CENTRAL": central_count,
            "NORTHERN": northern_count
        }
        
        inferred = max(counts, key=counts.get)
        
        # Calculate winning probability with smoothing (bound between 0.60 and 0.98)
        winner_count = counts[inferred]
        probability = 0.60 + 0.38 * (winner_count / total_regional_hits)
        
        logger.info(
            f"Dialect classification: Southern={southern_count}, Central={central_count}, "
            f"Northern={northern_count}. Inferred={inferred} (p={probability:.2f})"
        )

        return {
            "inferred": inferred,
            "probability": round(probability, 2)
        }
