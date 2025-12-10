# src/core/dspy_qa.py

from pathlib import Path
from typing import Type
from src.core.processor import QAPromptProcessor
import dspy
from src.utils.config import config

processor = QAPromptProcessor(config)

def load_prompt_template(template_path: str) -> str:
    """Load a prompt template from a text file."""
    path = Path(template_path)
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def build_qa_signature_from_template(template_path: str) -> Type[dspy.Signature]:
    """
    Dynamically build a DSPy Signature for QA generation.

    - The class docstring is set to the template content.
    - Inputs:
        * context: big user document
        * num_questions: how many Q&A pairs to generate
    - Output:
        * qa_json: JSON string of Q&A pairs
    """
    raw_prompt_text = load_prompt_template(template_path)
    prompt_text = processor.render(raw_prompt_text)


    class QAGenerationSignature(dspy.Signature):
        """This will be overwritten with the template content."""
        
        context = dspy.InputField(
            desc="input your conext to generate questions and answers"
        )
        # JSON output
        qa_json: str = dspy.OutputField(
            desc="A JSON array of objects, each with 'question' and 'answer' fields."
        )

    # Inject your template as the docstring used by DSPy
    QAGenerationSignature.__doc__ = prompt_text

    return QAGenerationSignature
