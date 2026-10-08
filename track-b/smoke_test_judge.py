import sys
sys.path.append('.')
from src.judge import check_faithfulness

source = "The agency spent 5 million dollars on new software in 2023. No layoffs occurred."
summary = "The agency spent 5 million dollars on software. The agency also hired 200 new employees."
cfg = {"temperature": 0.2, "max_tokens": 500}

result = check_faithfulness(source, summary, cfg)
print(result)