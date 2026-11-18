# Gemma-2-9B n=100 expansion

Run the five model notebooks in fresh Kaggle GPU drafts, then run the postprocess notebook.
All use the fixed seed-42 positions 25:125 split, 1000 paired bootstrap draws, and
greedy juice_coherent decoding with 16 new tokens. INT4 notebooks require the existing
Gemma N_CALIB=128 checkpoint as an attached Notebook Output. Do not edit the paper until
`gemma_n100_raw_results.json` has been reviewed.
