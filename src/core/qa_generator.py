# src/core/qa_generator.py

import os
from typing import Dict, Any, Optional
from src.core.providers.dspy_provider import DspyProvider
from src.utils.logger import logger


class QAGenerator:
    def __init__(self, config: Dict[str, Any]):
        """
        High-level wrapper around DspyProvider for QA generation.

        Expected config shape (example):

        {
            "provider": {
                "model": "ollama/gemma3:4b",
                "api_key": "...",                     # if needed
                "lm_kwargs": {"max_tokens": 4096},
                "qa_template_path": "src/templates/qa_template.txt"
            },
            "context_dir": "src/context",
            "default_num_questions": 10
        }
        """
        self.config: Dict[str, Any] = config or {}

        # Where context .txt files live (user uploads)
        self.context_dir: str = self.config.get("context_dir", "src/context")

        # Provider-specific config (fallback: treat whole config as provider config)
        provider_config: Dict[str, Any] = self.config.get("provider", self.config)

        # Default number of Q&A pairs if not specified at call-time
        self.default_num_questions: int = self.config.get(
            "default_num_questions", 10
        )

        # Initialize DSPy-backed provider
        self.provider = DspyProvider(provider_config)

    def generate(
        self,
        context_filename: str,
        num_questions: Optional[int] = None,
    ) -> str:
        """
        Load context from a .txt file and generate QA JSON.

        Args:
            context_filename: Name of the .txt file inside `context_dir`
                              (e.g. "agency_123_context.txt").
            num_questions: Optional override. If None, uses default_num_questions.

        Returns:
            JSON string of Q&A pairs produced by DspyProvider.generate().
        """
        # Resolve full path to context file
        context_path = os.path.join(self.context_dir, context_filename)

        if not os.path.exists(context_path):
            logger.error(f"Context file not found: {context_path}")
            raise FileNotFoundError(f"Context file not found: {context_path}")

        # Read context text
        with open(context_path, "r", encoding="utf-8") as f:
            context_text = f.read()

        # Decide how many questions to ask for
        n_questions = num_questions or self.default_num_questions

        # Options passed into DspyProvider.generate()
        options = {"num_questions": n_questions}

        logger.info(
            f"Generating Q&A from context file='{context_path}' "
            f"with num_questions={n_questions} using model={self.provider.model_name}"
        )

        # Delegate to DspyProvider
        qa_json = self.provider.generate(context=context_text, options=options)

        return qa_json
