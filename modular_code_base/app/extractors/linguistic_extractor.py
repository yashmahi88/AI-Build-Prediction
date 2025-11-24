# import re
# from typing import List, Dict
# from app.extractors.base_extractor import BaseExtractor

# class LinguisticExtractor(BaseExtractor):
#     """Extract rules from linguistic patterns"""
    
#     def extract(self, content: str, metadata: Dict = None) -> List[Dict]:
#         rules = []
#         lines = content.split('\n')
        
#         patterns = [
#             (r'\bmust\b', 0.95),
#             (r'\bshall\b', 0.90),
#             (r'\brequired?\b', 0.85),
#             (r'\bshould\b', 0.75),
#             (r'\brecommended\b', 0.70),
#         ]
        
#         for line in lines:
#             line_stripped = line.strip()
#             if 10 < len(line_stripped) < 200:
#                 for pattern, conf in patterns:
#                     if re.search(pattern, line_stripped.lower()):
#                         rules.append(self.format_rule(line_stripped, conf, 'LINGUISTIC'))
#                         break
        
#         return rules
