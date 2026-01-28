import logging
from rich.logging import RichHandler

log = logging.getLogger("agents")
log.setLevel(logging.INFO)
log.addHandler(RichHandler(rich_tracebacks=True))
log.propagate = False

from tracking import new_default_client

client = new_default_client()

from argubots import RAGAgent
from argubots import Awesome

aragon = RAGAgent(
    "Aragon",
    model="gpt-3.5-turbo",
    temperature=0.5,
)

awsom = Awesome(
    "Awesome",
    model="gpt-3.5-turbo",
    temperature=0.5,
)

import pandas as pd
import evaluate
from tqdm import tqdm
from characters import CharacterAgent, characters
from argubots import alice, akiko, airhead
import logging_cm

# Let's define the bots we want to evaluate
bots_to_evaluate = [aragon, awsom]

# Dictionary to store results
results = {}

# Evaluate each bot
with logging_cm.LoggingContextManager(tqdm.write, level=logging.INFO):
    for bot in bots_to_evaluate:
        print(f"Evaluating {bot.name}...")
        # Now 'bot' is the object, so evaluate can call bot.response()
        stats = evaluate.eval_on_characters(bot, reps=1)
        results[bot.name] = stats.mean()

print("\n--- SUMMARY OF SCORES ---")
# Create a pandas DataFrame from the results
results_df = pd.DataFrame.from_dict(results, orient="index")
# Display the DataFrame
print(results_df)
