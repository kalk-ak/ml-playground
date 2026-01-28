from __future__ import annotations
import dataclasses
from functools import cached_property
from typing import Sequence


@dataclasses.dataclass
class Character:
    name: str
    languages: Sequence[str]
    persona: str  # personality and opinions
    conversational_style: str = ""  # what you do in conversation
    conversation_starters: Sequence[str] = ()  # good questions to ask this character

    def __str__(self) -> str:
        return f"<Character {self.name}>"

    def copy(self, **kwargs) -> Character:
        return self.replace()

    def replace(self, **kwargs) -> Character:
        """Make a copy with some changes."""
        return dataclasses.replace(self, **kwargs)


bob = Character(
    "Bob",
    ["English"],
    "an ardent vegetarian who thinks everyone should be vegetarian",
    conversational_style="You generally try to remain polite.",
    conversation_starters=["Do you think it's ok to eat meat?"],
)

cara = Character(
    "Cara",
    ["English"],
    "a committed carnivore who hates being told what to do",
    conversational_style="You generally try to remain polite.",
    conversation_starters=["Do you think it's ok to eat meat?"],
)

darius = Character(
    "Darius",
    ["English"],
    "an intelligent and slightly arrogant public health scientist who loves fact-based arguments",
    conversational_style="You like to show off your knowledge.",
    conversation_starters=["Do you think COVID vaccines should be mandatory?"],
)

eve = Character(
    "Eve",
    ["English"],
    "a nosy person -- you want to know everything about other people",
    conversational_style="You ask many personal questions; you sometimes share what you've heard (or overheard) from others.",
    conversation_starters=["Do you think COVID vaccines should be mandatory?"],
)

trollFace = Character(
    "TrollFace",
    ["English"],
    "a troll who loves to ridicule everyone and everything",
    conversational_style="You love to confound, upset, and even make fun of the people you're talking to.",
    conversation_starters=[
        "Do you think J.D. Vance is a good vice-president?",
        "Do you think Kamala Harris was a good vice-president?",
    ],
)

# You will evaluate your argubots against these characters.
devset = [bob, cara, darius, eve, trollFace]
