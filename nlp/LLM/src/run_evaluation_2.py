bot_to_evaluate = argubots.awsome

# the set of characters to test against
devset = characters.devset

# a dictionary to store the evaluation results
evaluation_summary = {}

print(f"--- Starting Evaluation for {bot_to_evaluate.name} ---")

# run the evaluation
evaluation_summary = evaluate.eval_on_characters(
    argubot=bot_to_evaluate,
    chars=devset,  # The standard development set of characters
    reps=1,  # How many times to test against each character
    turns=3,  # The number of turns in each dialogue
)

print(f"\n\n--- Evaluation Summary for {bot_to_evaluate.name} ---")

rich.print(evaluation_summary)
print("\nScores:")
rich.print(evaluation_summary.mean())
print("\nStandard Deviations:")
rich.print(evaluation_summary.sd())

print("\nEvaluation complete.")
