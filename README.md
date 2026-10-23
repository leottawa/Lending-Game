# Lending Game — HEC MSc DSAIB, Introduction to Finance for Data Scientists

We are **Lender 2**. `stage1_pricing.py` is the team's Stage 1 script (fintech name: "Lender 2: Judgment Day"). It fits a
spline GAM default model on all three signals, integrates over competitors' unseen signals and pricing
behaviour to correct for the winner's curse, and offers the profit-maximising rate per applicant.

## Run Stage 1

1. Put the two course files next to the script, keeping their names:
   - `PastLoans.csv`
   - `NewApplications_Lender2_Round1.csv`
2. Run it (under a minute on a laptop):

   ```bash
   python3 stage1_pricing.py
   ```

3. It writes `Lender2_Round1_offers.csv` with columns `id` and `rate` (0.12 = 12%, empty = no offer).

Data files and submissions are kept out of git by `.gitignore`.

## Stage 1 deliverables (due 17 October, 24:00, by email to hombert@hec.fr)

- The fintech's name.
- `Lender2_Round1_offers.csv`.

## Stage 2 reminder

The script reads `NewApplications_Lender2_Round1.csv` by name. For Round 2, change `NEW_FILE`
(and `OUT_FILE`) at the top of the script, otherwise it silently reuses the Round 1 file.
