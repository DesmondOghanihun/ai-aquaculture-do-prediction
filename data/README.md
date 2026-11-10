@'
# Dataset Description

`water_quality_data_Mar_Aug_2026.csv` contains 1,422 water-quality observations
across 18 commercial fish ponds in Delta State, Nigeria, sampled thrice weekly
between 2026-03-02 and 2026-08-31.

## Pond types
- Tarpaulin (6 ponds, Okwe and Ibusa)
- Concrete (6 ponds, Issele Asagba and Okpanam)
- Earthen (6 ponds, Anwai and Ugbolu)

## Variables
See manuscript Section 3.1 for the full list.

## Target variable
`DO_Risk` = 1 if `DO_mgL` < 4.0, else 0 (12.9% positive class).

## Licence
CC-BY-4.0 — free to reuse with attribution.

## Citation
See the root README.md.
'@ | Out-File -Encoding utf8 data\README.md

git add data/README.md