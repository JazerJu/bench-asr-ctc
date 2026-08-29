# coding: utf-8
"""Hotword phoneme matching pipeline for bench-asr-ctc."""

import logging

logger = logging.getLogger("bench_asr_ctc.hotword")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[hotword %(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

from .hot_phoneme import PhonemeCorrector, CorrectionResult

__all__ = [
    "PhonemeCorrector",
    "CorrectionResult",
    "logger",
]
