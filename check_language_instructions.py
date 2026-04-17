
#!/usr/bin/env python3
"""
Check the language instructions in our training data vs test data.
"""

import json
import random

# Let's check a few things
print("=" * 80)
print("CHECKING LANGUAGE INSTRUCTION CONSISTENCY")
print("=" * 80)
print()

# First, let's check what task instructions genie_sim used
print("1. GenieSim test instructions (from benchmark):")
print("-" * 80)
test_instructions = [
    "Grab the yellow package on the table, turn the waist right to face the barcode scanner, place the package on the scanning table with the barcode facing up. Then, grab the package, rotate the waist and place the package in the blue bin. Finally, return the waist back to face the initial table",
    "Grab the black package on the table, turn the waist right to face the barcode scanner, place the package on the scanning table with the barcode facing up. Then, grab the package, rotate the waist and place the package in the blue bin. Finally, return the waist back to face the initial table"
]
for instr in test_instructions:
    print(f"- {instr}")
print()

# Now let's check what ACoT-VLA used for training
print("2. ACoT-VLA training instructions (from config):")
print("-" * 80)
acot_instructions = {
    "Sort packages": (
        "Grab the <color> package on the table, "
        "turn the waist right to face the barcode scanner, "
        "place the package on the scanning table with the barcode facing up. "
        "Then, grab the package, "
        "rotate the waist and place the package in the blue bin. "
        "Finally, return the waist back to face the initial table",
        0.5
    )
}
for task, (instr, prob) in acot_instructions.items():
    print(f"Task '{task}':")
    print(f"  - {instr}")
print()

# Now let's check what UnifoLM-VLA's prompt looks like
print("3. UnifoLM-VLA inference prompt:")
print("-" * 80)
print("The task is \"<instruction>\".")
print()

print("4. UnifoLM-VLA TRAINING prompt (from RLDSBatchTransform):")
print("-" * 80)
print("You are a robot using the joint control. The task is \"<instruction>\". Please predict up to 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point.")
print()

print("=" * 80)
print("POTENTIAL ISSUE #1: Prompt mismatch!")
print("=" * 80)
print("- Training: Complex prompt asking for trajectory points as (x,y) tuples")
print("- Inference: Simple prompt \"The task is ...\"")
print()

print("=" * 80)
print("POTENTIAL ISSUE #2: Let's check the actual model inference code!")
print("=" * 80)
