# CPI tables (verified 2026-07-07)

Used only for the absolute-dollar secondary analyses and the L6 total-burden
ledger; the primary analysis is within-campus percentile ranks (currency-free).
Deflation: cost_2021 = cost_nominal * (CPI_2021 / CPI_year), within country.

## us_cpiu_annual.csv

US CPI-U, all items, U.S. city average, annual averages, 1982-84=100.
Source of record: U.S. Bureau of Labor Statistics historical CPI-U tables
(https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202402.pdf;
bls.gov blocks automated fetches, HTTP 403). Values extracted 2026-07-07 from
the BLS-mirroring table at
https://www.usinflationcalculator.com/inflation/consumer-price-index-and-annual-percent-changes-from-1913-to-2008/
and cross-checked against the BLS-published 2021 annual average (270.970).

## canada_cpi_annual.csv

Canada CPI, all-items, annual average, not seasonally adjusted, 2002=100.
Source: Statistics Canada table 18-10-0005-01, retrieved programmatically
2026-07-07 via the official Web Data Service
(POST https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods,
vector v41693271, productId 18100005). Raw API response preserved in
statcan_18100005_raw_response.json.
