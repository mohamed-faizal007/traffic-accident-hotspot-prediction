# Five-minute demo script

A click-by-click walkthrough of the dashboard that shows the project solving its stated problem: telling a traffic authority **where and when** to act next month, instead of showing a static heatmap of the past.
Every number below is what the UI shows; each comes from the saved results (`results/`, `outputs/`) and was checked against them.

## Before you start

1. From the project root: `.\run-web.ps1`, then open <http://localhost:5173>. Stop it afterwards with Ctrl+C.
2. The maps need internet for their base tiles (the Folium map also for its scripts). If the connection fails, the coloured points still appear and
   [`docs/screenshots/`](screenshots/) holds a screenshot of every page to fall back on.
3. Use a 1280 px-wide window or larger. The page works down to phone width, but the tables and charts read best wide.

## The script

**0:00 Home.** "The question is: which 500 m cells of Great Britain are likely to have at least two recorded collisions *next month*?"
Point at the four cards: on the 2025 test year (evaluated once, with a configuration frozen beforehand) average precision is **0.255, which is 10.9 times the 2.34% base rate**.
The top 1% of cells each month contain hotspots **43.7%** of the time (18.6 times the base rate); simply ranking by the last 12 months' collisions gives **42.2%**, and average precision 0.220.
Say the honest framing now: "a well-chosen model beats the simple baseline by +0.035 average precision, but only modestly." Point at the amber card: only cells with 3 or more collisions in 2021 to 2023 are modelled, which caps recall at **91.7%**.

**0:50 Action list** (nav: Action list). "This is what a planner would use." The page title says *cells to prioritise in 2026-01* and the amber note says **unvalidated forecast (no outcomes yet)**.
Tiers default to Critical and High: **1,520 of 30,388 modelled cells**. Row 1 is cell **1059_362**: Critical, **84.3%** estimated probability, **3.70** predicted collisions, and a 12-month sparkline.
Point at the column notes: the predicted count carries the "unvalidated" label, and "What stands out" is labelled *descriptive, not a model attribution*.

**1:40 Cell detail.** Click row 1. The panel shows probability 84.3%, XGBoost 3.70 against the historical-average 3.50, a larger monthly chart (peak 11 in 2025-10), and the facts behind the ranking:
**54 collisions in the last 12 months**, more than 99% of modelled cells; a long-run average of 3.75 per month; 4 collisions in the latest month. The road and weather profile reads "single carriageway", "mostly fine weather".
Say: "These are facts about the cell's own history, not a model explanation. History is what the model uses most: in permutation importance one history feature, the long-run average, accounts for 0.2075 of the drop in average precision; the next feature is 0.0028." (Feature Importance page.)

**2:10 Filters, then the map** (nav: Risk Map). The forecast shows the 3,039 Critical, High and Medium cells. Click **Single carriageway**, then **Adverse-weather share: top 10% of cells**.
The status line reads: *37 cells match (adverse-weather share ≥ 0.40, the 90th percentile of modelled cells)*, and the map shows those 37. Mention the note: the Folium map uses a different, stricter cut-off, so it tags far fewer cells as "adverse".
Click **Open the Folium map** (new tab): the original interactive map with the same tiers, popups and its own road/weather tag filter.

**3:00 Patterns, temporal tab** (nav: Patterns). The heatmap: collisions peak in the **17:00 hour (8.7% of all collisions)**, and **Friday** has the most of any weekday (16.6%). The weekday peak hour is 17:00; the weekend peak hour is 13:00.
A weekend day averages **0.83 times** the collisions of a weekday. Say these patterns are descriptive: the frozen classifier does not use them; only the regression track has temporal features.

**3:25 Patterns, DBSCAN clusters tab.** "DBSCAN, fitted on 2021 to 2023 only, finds **2,161 clusters** that cover **16.9%** of collisions; the largest holds 1,162 collisions (2.2% of clustered points)."
Drag the slider or click the *95th percentile or larger* preset: **109 clusters** remain. Show the sweep table with the chosen row (eps 0.10 km, min 10) and the selection criterion.
Moran's I is **0.496** (p = 0.002): neighbouring cells have similar counts, so treating cells as independent understates uncertainty. Then read the blue note: **cluster features did not improve prediction**.

**4:00 Regression track** (nav: Regression track). Model comparison: on the 2025 test set XGBoost's MAE is **0.2875 against the baseline's 0.2844** (a gap of +0.0031, +1.1%), while its RMSE (0.4660 against 0.5126) and R² (0.0377 against -0.1642) are better.
On validation the MAE gap was -0.0438 (-15.5%), so the direction differed between validation and test. "We report that as it is: the test set was used once."
Scroll to **Quantile tiers**: observed hotspot rate is 0.31% (LOW), 0.66% (MEDIUM) and 6.06% (HIGH); the rank tiers separate more sharply, from 43.72% (Critical) down to 1.01% (Low).

**4:35 Ablations** (same page, *Ablations*). Three additions were tested and none helped: DBSCAN cluster features lowered validation average precision by 0.0030 (0 of 5 seeds better); neighbouring-cell features made no distinguishable difference on the classifier (+0.0010, interval -0.0009 to +0.0019) and worsened the regression MAE by +0.0038 (interval +0.0036 to +0.0040).
Close with: "A cell's own recent history carries almost all of the signal; the dashboard says so plainly, and the limitations are on the Home page."

## If something goes wrong

- A page shows "Could not load …: <message>": the message names the missing file and the script that regenerates it. Other sections still work.
- No map background: the base-map tiles need the internet; the cells still show. Use the screenshots in `docs/screenshots/`.
- Ports 8000 or 5173 busy: stop the earlier `run-web.ps1` (Ctrl+C), or free the port.

## Likely examiner questions

**1. Why is a classifier the primary system and not the count regression?**
The question a planner asks is "which cells should we look at first?", which is a ranking and prioritisation task. The classifier predicts whether a cell reaches at least two collisions next month and is judged by precision-recall: on 2025 it reaches average precision 0.255 (10.9 times the 2.34% base rate) and beats the trailing-12-month baseline by +0.035 (95% grid-bootstrap CI +0.031 to +0.038).
The proposal's regression (count against a historical-average baseline) was built and evaluated with the same discipline as an additive track. The count is hard to predict: about 85% of 2025 cell-months have zero collisions, and test R² is only 0.0377.

**2. Why is the regression's MAE tied with the baseline?**
On the 2025 test set XGBoost's MAE is 0.2875 and the baseline's 0.2844 (+0.0031, +1.1%), although its RMSE and R² are better. On 2024 validation XGBoost looked clearly better (MAE gap -0.0438, -15.5%). With most cells at zero, MAE rewards predicting zero; RMSE and R² reward getting the larger counts roughly right.
The test set was evaluated once under a frozen configuration, so we report the gap rather than tune on it.

**3. Why were the DBSCAN features dropped?**
They were tested as predictors on the 2024 validation set against a rule written beforehand: average precision fell from 0.2510 to 0.2480 (-0.0030, worse in 5 of 5 seeds, interval -0.0049 to -0.0021). The chosen clusters (eps 0.1 km, min 10; 2,161 of them) cover only 16.9% of training collisions.
DBSCAN is still used to describe where collisions concentrate. Why the features do not help is an untested hypothesis (for training rows they are built from the same years that produce the labels).

**4. What is the 91.7% coverage ceiling?**
Only cells with at least 3 collisions in 2021 to 2023 are modelled (30,388 cells), so that no 2024 or 2025 information leaks into which cells are included. Those cells contain 8,549 of the 9,320 hotspot cell-months in 2025 (91.7%).
System-wide recall can therefore not exceed 91.7%: at the frozen threshold it is 0.330 against 0.360 on modelled cells.

**5. What is the January 2026 forecast, and what is it not?**
It is the frozen classifier (and, for counts, the frozen regression model, run once for inference only with its file hash checked before and after) applied to data through December 2025. The classifier's probabilities are calibrated on 2024.
It has no outcomes yet, so it is **unvalidated**, and the dashboard says so wherever it appears. It is a ranking of where recorded collisions are likely to cluster, not a statement of cause, and not a guarantee. It can be checked as soon as January 2026 collisions are published.

**6. What are the "drivers" in the Action list, and what are they not?**
They are descriptive facts read from the cell's own history: collisions in the last 12 months, the long-run monthly average, months since the last collision, the busiest month, with percentiles computed among the modelled cells. They explain why a cell looks risky.
They are **not** a model attribution: no SHAP values or coefficients are involved, and they do not say how much each fact changed the model's score. The model-level view of what matters is the permutation importance on the Feature Importance page.
