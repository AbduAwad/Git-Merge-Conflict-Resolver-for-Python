# coding: utf8
from __future__ import unicode_literals

<<<<<<<<< Temporary merge branch 1
from .pipes import Tagger, DependencyParser, EntityRecognizer  # noqa
from .pipes import TextCategorizer, Tensorizer, Pipe  # noqa
from .morphologizer import Morphologizer
from .entityruler import EntityRuler  # noqa
from .hooks import SentenceSegmenter, SimilarityHook  # noqa
from .functions import merge_entities, merge_noun_chunks, merge_subtokens  # noqa
=========
from .pipes import Tagger, DependencyParser, EntityRecognizer
from .pipes import TextCategorizer, Tensorizer, Pipe
from .entityruler import EntityRuler
from .hooks import SentenceSegmenter, SimilarityHook
from .functions import merge_entities, merge_noun_chunks, merge_subtokens

__all__ = [
    "Tagger",
    "DependencyParser",
    "EntityRecognizer",
    "TextCategorizer",
    "Tensorizer",
    "Pipe",
    "EntityRuler",
    "SentenceSegmenter",
    "SimilarityHook",
    "merge_entities",
    "merge_noun_chunks",
    "merge_subtokens",
]
>>>>>>>>> Temporary merge branch 2
