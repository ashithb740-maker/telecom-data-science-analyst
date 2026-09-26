# Virtual Data Science with Python — Week 1

## Project
Data Acquisition, Cleaning and Preprocessing of the UCI Online Retail Dataset Using Python.

## Dataset
UCI Online Retail Dataset. Source: UCI Machine Learning Repository.

## Initial dataset
- Records: 541,909
- Columns: 8
- Duplicate rows: 5,268
- Missing CustomerID: 135,080
- Negative Quantity records: 10,624
- Non-positive UnitPrice records: 2,517
- Cancellation invoices: 9,288

## Cleaning
Exact duplicates, cancellations, non-positive quantities/prices, missing CustomerID/Description, and invalid dates were handled. A TotalAmount feature was created. IQR was used to identify unusually high/low transaction values; legitimate high-value transactions were flagged rather than automatically deleted.

## Output
Cleaned data: `data/processed/online_retail_cleaned.csv`

## Week 1
This repository contains the Week 1 preprocessing script, report, figures, and processed dataset for the Virtual Data Science with Python Trainee internship.
