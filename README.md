# ERA5 Climate Trend Analysis

Long-term temperature and precipitation trend analysis using ERA5 reanalysis data, with statistical significance testing and anomaly detection.

**Author:** Goutam Raman · MSc Integrated Climate System Sciences, Universität Hamburg

---

## What it does

- Analyses 40+ years of monthly climate data (1980–2023)
- Detects warming trends using linear regression with p-value significance testing
- Computes temperature anomalies against the 1981–2010 WMO baseline
- Analyses monsoon season precipitation trends (JJAS)
- Counts extreme heat and rainfall events per decade
- Generates a 6-panel publication-quality report

## Key findings (Kerala, India)

| Metric | Value |
|--------|-------|
| Warming trend | +0.25°C / decade |
| Total warming (1980–2023) | +1.08°C |
| Monsoon precipitation trend | +64 mm / decade |
| Trend significance | p < 0.001 |

## Sample output

![ERA5 Climate Report](outputs/era5_climate_report.png)

## Quickstart

```bash
pip install numpy pandas matplotlib scipy
python era5_analysis.py
```

## Using real ERA5 data

Replace `generate_synthetic_era5()` with a Copernicus CDS API download:

```python
import cdsapi
c = cdsapi.Client()
c.retrieve('reanalysis-era5-single-levels-monthly-means', {
    'variable': ['2m_temperature', 'total_precipitation'],
    'year': [str(y) for y in range(1980, 2024)],
    'month': [f'{m:02d}' for m in range(1, 13)],
    'area': [13, 74, 8, 78],  # Kerala bounding box
    'format': 'netcdf',
}, 'kerala_era5.nc')
```

Register free at: https://cds.climate.copernicus.eu

## Methods

- Trend detection: Ordinary Least Squares linear regression
- Anomaly baseline: 1981–2010 (WMO standard reference period)
- Extreme event threshold: 95th percentile of full record
- ENSO / IOD variability modelled as sinusoidal oscillations
