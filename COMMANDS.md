# Makri Commands Guide

## Core Commands

- train
  - Train on all .txt files in datas/ (parallel)
  - Example: cargo run --release -- train --bits=64
  - Optional: --normalize-data (equalize per-file influence)

- generate
  - Generate text from the current model
  - Example: cargo run --release -- generate 80 --bits=64 --pass "He would sit on a rock, and " --stream
  - If nudge_bot_model.frozen exists, it is used automatically

- talk
  - Interactive chat loop
  - Example: cargo run --release -- talk 40 --bits=64

- train-fragment
  - Train a single file and save as a fragment
  - Example: cargo run --release -- train-fragment datas/shake.txt --bits=64

- merge-fragments
  - Merge all fragments into a single model
  - Example: cargo run --release -- merge-fragments --bits=64

- freeze
  - Build a frozen, index-based model (nudge_bot_model.frozen)
  - Example: cargo run --release -- freeze --bits=64

## Flags

- --bits=8
  - 8-bit quantized weights (fastest)

- --bits=64
  - 64-bit f32 weights

- --bits=128
  - 128-bit f64 weights

- --bits=256
  - TRUE 128-bit u128 weights

- --normalize-data
  - Equalize per-file influence during training

- --pure
  - Pure Markov (no context). Kept for comparison only.

- --pass
  - Provide a starting prompt for generation

- --stream
  - Print tokens as they are generated
