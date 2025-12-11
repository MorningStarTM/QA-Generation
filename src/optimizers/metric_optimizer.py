from src.eval.information import WeightedInformationCoverageMetric
from src.eval.similarity import QADiversityMetric
from src.utils.config import config
import dspy
import json
from src.utils.utils import qa_to_json

div_metric = QADiversityMetric(embedding_model=config.get("embedding_model","all-MiniLM-L6-v2"))
info_metric = WeightedInformationCoverageMetric(embedding_model=config.get("embedding_model","all-MiniLM-L6-v2"))


def composite_metric(generated_text, topics, trace=None):
   qa2json = qa_to_json(generated_text.completions)
   gen_text = json.dumps(qa2json, ensure_ascii=False)

   score1 = div_metric(qa2json, trace)

   score2 = info_metric(gen_text, topics, trace)
   score = (score1 + score2) / 2.0
   feedback = f"You scored {score1}/1.0 and {score2}/1.0 on diversity and information coverage, respectively"
   return dspy.Prediction(score=score, feedback=feedback)