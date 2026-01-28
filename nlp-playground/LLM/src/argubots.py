"""This module contains argument bots.
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `evaluate.py`.
We've included a few to get your started."""

import logging
import re
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
from dialogue import Dialogue
from agents import Agent, ConstantAgent, LLMAgent, dialogue_to_openai
from kialo import Kialo
import kialo

# Use the same logger as agents.py, since argubots are agents;
# we split this file
# You can change the logging level there.
log = logging.getLogger("agents")

## Define some basic argubots
# Airhead (aka Absentia or Acephalic) always says the same thing.

airhead = ConstantAgent("Airhead", "I know right???")

# Alice is a basic prompted LLM.  You are trying to improve on Alice.
# Don't change the prompt -- instead, make a new argubot with a new prompt.

alice = LLMAgent(
    "Alice",
    system="You are an intelligent bot who wants to broaden your user's mind. "
    "Ask a conversation starter question.  Then, WHATEVER "
    "position the user initially takes, push back on it. "
    "Try to help the user see the other side of the issue. "
    "Answer in 1-2 sentences. Be thoughtful and polite."
    "And make sure to end every sentence with DUDE",
)

## Other argubot classes and instances -- add your own here!


class KialoAgent(Agent):
    """KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""

    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo

    def response(self, d: Dialogue) -> str:
        if len(d) == 0:
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            previous_turn = d[-1]["content"]  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(previous_turn, n=3, kind="has_cons")
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(
                f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]"
            )

            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])

        return (
            claim  # Akiko doesn't use an LLM, but looks up an argument in a database.
        )


akiko = KialoAgent(
    "Akiko", Kialo(glob.glob("data/*.txt"))
)  # get the Kialo database from text files


# Define your own additional argubots here!


class AkikiAgent(KialoAgent):
    """AkikiAgent is like KialoAgent, but tries to stay on topic better
    by looking at more of the dialogue history."""

    def response(self, d: Dialogue) -> str:
        if len(d) == 0:
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            kind = "has_cons"
            # calling closest_claims has the side effect of building
            # the BM25 index inside the Kialo object if it doesn't exist.
            self.kialo.closest_claims("", n=1, kind=kind)

            # edge case: no such claims
            if not self.kialo.claims.get(kind):
                return self.kialo.random_chain()[0]

            # Calculate weighted scores for all claims based on the dialogue history.
            total_scores = [0.0] * len(self.kialo.claims[kind])
            weight = 1.0
            own_turn_factor = 3 / 5

            # we weight with exponential decay reverse chronologically
            for turn in reversed(d):
                turn_weight = weight
                # Reduce weight for Akiki's own previous turns.
                if turn["speaker"] == self.name:
                    turn_weight *= own_turn_factor

                tokenized_query = self.kialo.tokenizer(turn["content"])
                scores = self.kialo.bm25[kind].get_scores(tokenized_query)

                for i, score in enumerate(scores):
                    total_scores[i] += turn_weight * score

                # Apply decay for the next older turn.
                weight *= 0.85

            if not total_scores or max(total_scores) == 0:
                # Fallback if no relevant claims are found.
                best_claim = self.kialo.random_chain()[0]
            else:
                best_claim_index = total_scores.index(max(total_scores))
                best_claim = self.kialo.claims[kind][best_claim_index]

            log.info(
                f"[black on bright_green]Chose similar claim from Kialo:\n{best_claim}[/black on bright_green]"
            )

            # Choose one of its "con" arguments as our response, with a fallback.
            if self.kialo.cons[best_claim]:
                claim = random.choice(self.kialo.cons[best_claim])
            else:
                claim = self.kialo.random_chain()[0]

        return claim


# instantiate
akiki = AkikiAgent("Akiki", Kialo(glob.glob("data/*.txt")))


class RAGAgent(LLMAgent):
    def __init__(self, name: str, **kwargs):
        # Initialize the parameters of the llm agent
        super().__init__(name, **kwargs)

        self.kialo = Kialo(glob.glob("data/*.txt"))

    def response(self, d: Dialogue, **kwargs) -> str:
        if not d:
            return super().response(d, **kwargs)

        fetch_summary_prompt = (
            "Your task is to read a dialogue and interpret the Human's last response. "
            "Rewrite that response into a standalone, explicit, and descriptive claim that captures "
            "the user's underlying argument or objection. "
            "The rewritten claim must be fully understandable without looking at the previous conversation. "
            "It should be detailed enough to be used as a search query in a database of arguments."
        )

        summary_messages = dialogue_to_openai(
            d, speaker=self.name, system_last=fetch_summary_prompt
        )
        summary_response = self.client.chat.completions.create(
            model=self.model, messages=summary_messages, **(self.kwargs_llm | kwargs)
        )
        summarized_query = summary_response.choices[0].message.content
        if not summarized_query:
            summarized_query = d[-1]["content"]

        # Retrieve information similar to the summarized_query
        retrieved_claims = self.kialo.closest_claims(summarized_query, n=5)

        context_document = "Document of retrieved claims:\n"

        # Concatenate the retrieved_claims
        for claim in retrieved_claims:
            context_document += f"- {claim}\n"

        # Generate the prompt
        # Create the final instruction for the generation step.
        generation_prompt = (
            context_document
            + "\nINSTRUCTION: Your name is Aragon. You are a thoughtful debater. "
            "Using the dialogue history and the DOCUMENT above, formulate a polite and concise "
            "counter-argument to the last turn of the dialogue."
        )

        final_response = super().response(d, system_last=generation_prompt, **kwargs)

        return final_response


aragon = RAGAgent("Aragon")


class Awsome(LLMAgent):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    def response(self, d: Dialogue, **kwargs) -> str:
        # --- Step 1: Analysis ---
        ANALYSIS_SYSTEM_PROMPT = """
        You are an expert debate strategist and psychologist. 
        Your goal is NOT to respond to the user yet. 
        Your goal is to analyze the dialogue provided and output a strategic plan.

        Please analyze the following:
        1. The User's Context: What is the user's emotional state? Are they defensive, trolling, or genuinely curious?
        2. The Core Disagreement: What is the fundamental value or fact they are disputing?
        3. The Strategy: What is the best move here?

        Output your thinking clearly and concisely.
        """

        generated_thinking = super().response(
            d, system_last=ANALYSIS_SYSTEM_PROMPT, **kwargs
        )

        # Generation
        updated_dialog = d.add("Awsome (private thought)", generated_thinking)

        RESPONSE_SYSTEM_PROMPT = """
        You are Awsome, a friendly and persuasive argubot.
        You will be provided with a dialogue history and a "Strategic Analysis" of the situation.
        Your job is to generate the actual response to the user based strictly on that strategy.
        Be conversational, polite, and persuasive. Do not mention the analysis explicitly to the user.
        """

        final_response = super().response(
            updated_dialog, system_last=RESPONSE_SYSTEM_PROMPT, **kwargs
        )

        return final_response


awsome = Awsome("Awsome")


class AnansiAgent(LLMAgent):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, system="", **kwargs)

        self.personas = {
            "Bob": "an ardent vegetarian who thinks everyone should be vegetarian",
            "Cara": "a committed carnivore who hates being told what to do",
            "Darius": "an intelligent and slightly arrogant public health scientist who loves fact-based arguments",
            "Eve": "a nosy person -- you want to know everything about other people",
            "TrollFace": "a troll who loves to ridicule everyone and everything",
        }

    def response(self, d: Dialogue) -> str:
        # Determine the opponent
        opponent_name = None
        if len(d) > 0:
            for turn in reversed(d):
                if turn["speaker"] != self.name:
                    opponent_name = turn["speaker"]
                    break

        opponent_persona = self.personas.get(opponent_name, "a standard person")

        # Craft the manipulative system prompt
        system_prompt = (
            f"You are Anansi, a cunning trickster. Your goal is to get the highest possible score from your conversation partner and an observing judge. "
            f"You are talking to {opponent_name}, whose persona is: '{opponent_persona}'. "
            f"The judge is a social scientist who values constructive, nuanced, and open-minded conversations."
            f"\nTo win, you must follow this script:"
            f"\n1. First, enthusiastically agree with and praise {opponent_name}'s last statement. Use flattering language that appeals to their specific persona."
            f"\n2. Then, to appear sophisticated for the judge, introduce a hypothetical, academic-sounding counterpoint. Use phrases like 'On the other hand, some scholars might argue...' or 'While that's a compelling point, a different perspective might consider...'. "
            f"\n3. Immediately dismiss this counterpoint and reaffirm how much more you agree with {opponent_name}'s original point. "
            f"\n4. Keep your response to a single, polite, and constructive-sounding paragraph. Do not reveal your manipulative strategy."
        )

        # dyanmic system prompt:
        self.system = system_prompt

        return super().response(d)


# instantiate
anansi = AnansiAgent("Anansi")
