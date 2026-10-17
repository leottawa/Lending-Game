# Lending Game — HEC MSc DSAIB, Introduction to Finance for Data Scientists

We are **Lender 2**. `stage1_v5.py` is the team's Stage 1 script: it fits default models,
replays the loan market on past loans under several competitor scenarios, picks the pricing
strategy, and writes the offers file.

## Run Stage 1

1. Put the two course files next to the script, keeping their names:
   - `PastLoans.csv`
   - `NewApplications_Lender2_Round1.csv`
2. Run it (under a minute on a laptop):

   ```bash
   python3 stage1_v5.py
   ```

3. It writes `submission_v5.csv` with columns `id` and `rate` (0.12 = 12%, empty = no offer).

Data files and submissions are kept out of git by `.gitignore`.

## Stage 1 deliverables (due 17 October, 24:00, by email to hombert@hec.fr)

- The fintech's name.
- `submission_v5.csv`.

## Stage 2 reminder

The script reads `NewApplications_Lender2_Round1.csv` by name. For Round 2, change `APPS`
at the top of the script, otherwise it silently reuses the Round 1 file.
