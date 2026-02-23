import json
import os

RESULTS_FILE = "experiments/multi_complexity_results.json"


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        return

    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    print(f"{'Task':<32} | {'Base (LLMzip)':<13} | {'GDT (Ratcliff)':<14} | {'GDT (Myers)':<13}")
    print("-" * 80)

    for r in results:
        name = r["name"]
        chars = r["target_chars"]

        # Calculate BPC
        base_bpc = r["standard_bits"] / chars
        gdt_rat_bpc = r["ctx_rat_bits"] / chars
        gdt_mye_bpc = r["ctx_mye_bits"] / chars

        print(
            f"{name:<32} | {base_bpc:<13.4f} | {gdt_rat_bpc:<14.4f} | {gdt_mye_bpc:<13.4f}")


if __name__ == "__main__":
    main()
