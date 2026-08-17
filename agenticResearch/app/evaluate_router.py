import pandas as pd
from model_router import SemanticRouter

router = SemanticRouter()

# Load evaluation dataset
CSV_PATH = r"D:\CODIN PLAYGROUND\ML-AI\research-Agent\agenticResearch\app\model_routing_eval_dataset.csv"

df = pd.read_csv(CSV_PATH)

# Evaluate router

results = []

for _, row in df.iterrows():
    task = row["task"]
    expected = row["expected_tier"]

    predicted_model = router.route(task)

    # Convert model name → tier
    if predicted_model == router.cheap:
        predicted_tier = "cheap"
    elif predicted_model == router.strong:
        predicted_tier = "strong"
    else:
        predicted_tier = "unknown"

    correct = predicted_tier == expected

    results.append({
        "task": task,
        "expected": expected,
        "predicted": predicted_tier,
        "model": predicted_model,
        "correct": correct,
    })


# Create results DataFrame
results_df = pd.DataFrame(results)

# Print individual results
for _, row in results_df.iterrows():

    status = "✓" if row["correct"] else "✗"

    print(
        f"\n{status} {row['task']}\n"
        f"   Expected : {row['expected']}\n"
        f"   Predicted: {row['predicted']}\n"
        f"   Model    : {row['model']}"
    )

# Calculate accuracy
accuracy = results_df["correct"].mean() * 100

# Summary
print("\n" + "=" * 60)
print("ROUTER EVALUATION")
print("=" * 60)

print(f"Total samples : {len(results_df)}")
print(f"Correct       : {results_df['correct'].sum()}")
print(f"Incorrect     : {(~results_df['correct']).sum()}")
print(f"Accuracy      : {accuracy:.2f}%")


# Breakdown by expected tier
print("\nAccuracy by tier:")

for tier in ["cheap", "strong"]:

    tier_df = results_df[results_df["expected"] == tier]

    if len(tier_df) > 0:
        tier_accuracy = tier_df["correct"].mean() * 100

        print(
            f"{tier:8} → "
            f"{tier_accuracy:.2f}% "
            f"({tier_df['correct'].sum()}/{len(tier_df)})"
        )


# Show incorrect predictions
incorrect_df = results_df[~results_df["correct"]]

if not incorrect_df.empty:

    print("\n" + "=" * 60)
    print("INCORRECT ROUTES")
    print("=" * 60)

    for _, row in incorrect_df.iterrows():

        print(
            f"\nTask: {row['task']}\n"
            f"Expected : {row['expected']}\n"
            f"Predicted: {row['predicted']}\n"
            f"Model    : {row['model']}"
        )