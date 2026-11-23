# Data Dictionary

The original notebook reads `cs-training.csv` and renames the original Give Me Some Credit columns as follows.

| Project feature | Original column | Description |
|---|---|---|
| `Target` | `SeriousDlqin2yrs` | 1 if serious delinquency occurred within two years; otherwise 0 |
| `CreditUtil` | `RevolvingUtilizationOfUnsecuredLines` | Total balance on credit cards and personal lines of credit, excluding real estate and installment debt, divided by the sum of credit limits |
| `Age` | `age` | Borrower's age in years |
| `DPD_30_59` | `NumberOfTime30-59DaysPastDueNotWorse` | Number of times 30–59 days past due (but no worse) |
| `DebtRatio` | `DebtRatio` | Monthly debt payments, alimony, and living costs divided by monthly gross income |
| `MonthlyIncome` | `MonthlyIncome` | Monthly income |
| `OpenCreditLines` | `NumberOfOpenCreditLinesAndLoans` | Number of open loans (installment such as car/mortgage) and lines of credit (e.g. credit cards) |
| `DPD_90plus` | `NumberOfTimes90DaysLate` | Number of times 90 or more days past due |
| `RealEstateLoans` | `NumberRealEstateLoansOrLines` | Number of mortgage and real-estate loans, including home equity lines of credit |
| `DPD_60_89` | `NumberOfTime60-89DaysPastDueNotWorse` | Number of times 60–89 days past due (but no worse) |
| `Dependents` | `NumberOfDependents` | Number of dependents in the family, excluding themselves |

The original dataset has missing values in `MonthlyIncome` and `Dependents`. The modeling notebook compares preprocessing pipelines and uses median imputation for income and most-frequent imputation for dependents in the tree-based model pipelines.
