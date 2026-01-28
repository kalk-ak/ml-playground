import characters
import evaluate

# the list of bots to evaluate
bots_to_evaluate = [
    argubots.alice,
    argubots.akiko,
    argubots.aragon,
]

# the set of characters to test against
devset = characters.devset

# a dictionary to store the evaluation results
evaluation_summaries = {}

print("--- Starting Evaluation ---")

with LoggingContext("evaluate"):
    for bot in bots_to_evaluate:
        print(f"Evaluating {bot.name}...")
        evaluation_summaries[bot.name] = evaluate.eval_on_characters(
            argubot=bot,
            chars=devset,  # The standard development set of characters
            reps=1,  # How many times to test against each character
            turns=3,  # The number of turns in each dialogue
        )

# Now, let's print out the results in a nicely formatted way.
print("\n\n--- Evaluation Summary ---")

for bot_name, summary in evaluation_summaries.items():
    print(f"\n--- {bot_name} ---")
    rich.print(summary)
    print("\nScores:")
    rich.print(summary.mean())
    print("\nStandard Deviations:")
    rich.print(summary.sd())

print("\nEvaluation complete.")
