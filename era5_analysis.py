"""
ERA5 Climate Data Analysis
===========================
Analyses long-term temperature and precipitation trends using ERA5-style
reanalysis data. Demonstrates statistical trend detection, anomaly
calculation, and climate visualisation.

In production: replace `generate_synthetic_era5()` with real ERA5 data
downloaded via the Copernicus CDS API (see era5_download.py).

Author: Goutam Raman
MSc Integrated Climate System Sciences, Universität Hamburg
Data source: ERA5 reanalysis (ECMWF / Copernicus Climate Change Service)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os
import json
from datetime import datetime

os.makedirs("outputs", exist_ok=True)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Synthetic ERA5-style data generator
# (replace with real CDS API download in production)
# ---------------------------------------------------------------------------

def generate_synthetic_era5(region: str = "Kerala, India",
                              start_year: int = 1980,
                              end_year: int = 2023) -> pd.DataFrame:
    """
    Generates realistic monthly climate data mimicking ERA5 reanalysis output.
    Includes long-term warming trend + natural variability + seasonal cycle.
    """
    years = range(start_year, end_year + 1)
    months = range(1, 13)
    records = []

    # Kerala climate baseline (tropical monsoon)
    temp_baseline = [26.5, 27.2, 28.8, 30.1, 29.8, 27.2,
                     26.5, 26.8, 27.0, 27.3, 27.0, 26.8]
    precip_baseline = [18, 22, 38, 115, 245, 390,
                       350, 290, 210, 310, 140, 40]

    warming_trend = 0.022   # ~0.022°C per year (consistent with observed India warming)
    precip_trend  = 1.8     # mm/year intensification

    for y in years:
        year_offset = y - start_year
        enso_effect = 0.6 * np.sin(2 * np.pi * year_offset / 3.8)   # ENSO ~3-8 yr cycle
        iod_effect  = 0.3 * np.sin(2 * np.pi * year_offset / 4.5)   # Indian Ocean Dipole

        for m in months:
            temp = (
                temp_baseline[m - 1]
                + warming_trend * year_offset
                + enso_effect * 0.4
                + np.random.normal(0, 0.5)
            )
            precip = max(0, (
                precip_baseline[m - 1]
                + precip_trend * year_offset * (1 if m in [6,7,8,9] else 0.2)
                + enso_effect * -25 * (1 if m in [6,7,8,9] else 0.1)
                + iod_effect * 15
                + np.random.normal(0, precip_baseline[m-1] * 0.18)
            ))
            records.append({
                "year": y, "month": m,
                "date": pd.Timestamp(year=y, month=m, day=15),
                "temperature_2m": round(temp, 3),
                "precipitation": round(precip, 2),
                "enso_index": round(enso_effect, 3),
            })

    df = pd.DataFrame(records)
    df["season"] = df["month"].map({
        12:"DJF", 1:"DJF", 2:"DJF",
        3:"MAM",  4:"MAM", 5:"MAM",
        6:"JJA",  7:"JJA", 8:"JJA",
        9:"SON", 10:"SON", 11:"SON"
    })
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9])
    df["region"] = region
    return df


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_anomalies(df: pd.DataFrame,
                       baseline_start: int = 1981,
                       baseline_end: int = 2010) -> pd.DataFrame:
    """Calculate temperature and precipitation anomalies vs. 1981-2010 baseline."""
    baseline = df[(df["year"] >= baseline_start) & (df["year"] <= baseline_end)]
    monthly_means = baseline.groupby("month")[["temperature_2m", "precipitation"]].mean()

    df = df.copy()
    df["temp_anomaly"] = df.apply(
        lambda r: r["temperature_2m"] - monthly_means.loc[r["month"], "temperature_2m"], axis=1)
    df["precip_anomaly"] = df.apply(
        lambda r: r["precipitation"] - monthly_means.loc[r["month"], "precipitation"], axis=1)
    return df


def compute_trends(df: pd.DataFrame) -> dict:
    """Linear trend analysis using Mann-Kendall-style OLS regression."""
    annual = df.groupby("year").agg(
        mean_temp=("temperature_2m", "mean"),
        total_precip=("precipitation", "sum"),
        mean_temp_anomaly=("temp_anomaly", "mean"),
    ).reset_index()

    t_slope, t_inter, t_r, t_p, t_se = stats.linregress(annual["year"], annual["mean_temp"])
    p_slope, p_inter, p_r, p_p, p_se = stats.linregress(annual["year"], annual["total_precip"])

    monsoon = df[df["is_monsoon"]].groupby("year")["precipitation"].sum().reset_index()
    m_slope, _, _, m_p, _ = stats.linregress(monsoon["year"], monsoon["precipitation"])

    return {
        "temperature": {
            "trend_per_decade": round(t_slope * 10, 3),
            "total_change": round(t_slope * (df["year"].max() - df["year"].min()), 2),
            "r_squared": round(t_r**2, 3),
            "p_value": round(t_p, 4),
            "significant": t_p < 0.05,
        },
        "precipitation": {
            "trend_per_decade_mm": round(p_slope * 10, 1),
            "r_squared": round(p_r**2, 3),
            "p_value": round(p_p, 4),
            "significant": p_p < 0.05,
        },
        "monsoon": {
            "trend_per_decade_mm": round(m_slope * 10, 1),
            "p_value": round(m_p, 4),
            "significant": m_p < 0.05,
        },
        "annual": annual,
        "monsoon_df": monsoon,
    }


def compute_extremes(df: pd.DataFrame) -> pd.DataFrame:
    """Count extreme events per decade."""
    df = df.copy()
    p95_temp  = df["temperature_2m"].quantile(0.95)
    p95_prec  = df["precipitation"].quantile(0.95)
    df["extreme_heat"]  = df["temperature_2m"] > p95_temp
    df["extreme_rain"]  = df["precipitation"]  > p95_prec
    df["decade"] = (df["year"] // 10) * 10
    return df.groupby("decade")[["extreme_heat", "extreme_rain"]].sum().reset_index()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_full_report(df: pd.DataFrame, trends: dict, region: str) -> str:
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#f8f9fa")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    NAVY   = "#1a3a5c"
    RED    = "#c0392b"
    BLUE   = "#2980b9"
    ORANGE = "#e67e22"
    GREEN  = "#27ae60"
    LGRAY  = "#ecf0f1"

    annual = trends["annual"]
    monsoon_df = trends["monsoon_df"]

    # ── 1. Annual temperature trend ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(LGRAY)
    ax1.bar(annual["year"], annual["mean_temp"], color=BLUE, alpha=0.5, width=0.8)
    slope = trends["temperature"]["trend_per_decade"] / 10
    trend_line = slope * annual["year"] + (annual["mean_temp"].iloc[0] - slope * annual["year"].iloc[0])
    ax1.plot(annual["year"], trend_line, color=RED, lw=2, label=f'+{trends["temperature"]["trend_per_decade"]}°C/decade')
    ax1.set_title("Annual mean temperature", fontsize=11, fontweight="bold", color=NAVY)
    ax1.set_ylabel("°C", fontsize=9)
    ax1.legend(fontsize=9)
    ax1.tick_params(labelsize=8)

    # ── 2. Temperature anomaly (warming stripes style) ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(LGRAY)
    anom_annual = df.groupby("year")["temp_anomaly"].mean().reset_index()
    colors = [RED if v >= 0 else BLUE for v in anom_annual["temp_anomaly"]]
    ax2.bar(anom_annual["year"], anom_annual["temp_anomaly"], color=colors, alpha=0.8, width=0.8)
    ax2.axhline(0, color="black", lw=0.8, linestyle="--")
    ax2.set_title("Temperature anomaly vs. 1981–2010 baseline", fontsize=11, fontweight="bold", color=NAVY)
    ax2.set_ylabel("°C anomaly", fontsize=9)
    ax2.tick_params(labelsize=8)

    # ── 3. Annual precipitation trend ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(LGRAY)
    ax3.bar(annual["year"], annual["total_precip"], color=BLUE, alpha=0.5, width=0.8)
    p_slope = trends["precipitation"]["trend_per_decade_mm"] / 10
    p_trend = p_slope * annual["year"] + (annual["total_precip"].iloc[0] - p_slope * annual["year"].iloc[0])
    ax3.plot(annual["year"], p_trend, color=ORANGE, lw=2,
             label=f'{trends["precipitation"]["trend_per_decade_mm"]:+.0f} mm/decade')
    ax3.set_title("Annual total precipitation", fontsize=11, fontweight="bold", color=NAVY)
    ax3.set_ylabel("mm/year", fontsize=9)
    ax3.legend(fontsize=9)
    ax3.tick_params(labelsize=8)

    # ── 4. Monsoon season precipitation ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(LGRAY)
    ax4.fill_between(monsoon_df["year"], monsoon_df["precipitation"],
                     alpha=0.4, color=GREEN)
    ax4.plot(monsoon_df["year"], monsoon_df["precipitation"], color=GREEN, lw=1.5)
    m_slope = trends["monsoon"]["trend_per_decade_mm"] / 10
    m_trend = m_slope * monsoon_df["year"] + \
              (monsoon_df["precipitation"].iloc[0] - m_slope * monsoon_df["year"].iloc[0])
    ax4.plot(monsoon_df["year"], m_trend, color=RED, lw=2, linestyle="--",
             label=f'{trends["monsoon"]["trend_per_decade_mm"]:+.0f} mm/decade (JJAS)')
    ax4.set_title("Monsoon season precipitation (Jun–Sep)", fontsize=11, fontweight="bold", color=NAVY)
    ax4.set_ylabel("mm", fontsize=9)
    ax4.legend(fontsize=9)
    ax4.tick_params(labelsize=8)

    # ── 5. Seasonal cycle ──
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor(LGRAY)
    recent  = df[df["year"] >= 2010].groupby("month")["temperature_2m"].mean()
    historic = df[df["year"] <= 1990].groupby("month")["temperature_2m"].mean()
    months_labels = ["J","F","M","A","M","J","J","A","S","O","N","D"]
    ax5.plot(range(1, 13), historic.values, "o--", color=BLUE, lw=1.5, label="1980–1990", markersize=5)
    ax5.plot(range(1, 13), recent.values,  "o-",  color=RED,  lw=2,   label="2010–2023", markersize=5)
    ax5.fill_between(range(1, 13), historic.values, recent.values, alpha=0.15, color=RED)
    ax5.set_xticks(range(1, 13))
    ax5.set_xticklabels(months_labels, fontsize=8)
    ax5.set_title("Seasonal temperature cycle — past vs. recent", fontsize=11, fontweight="bold", color=NAVY)
    ax5.set_ylabel("°C", fontsize=9)
    ax5.legend(fontsize=9)

    # ── 6. Extreme events ──
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor(LGRAY)
    extremes = compute_extremes(df)
    x = np.arange(len(extremes))
    w = 0.35
    ax6.bar(x - w/2, extremes["extreme_heat"], w, color=RED,   alpha=0.8, label="Extreme heat months")
    ax6.bar(x + w/2, extremes["extreme_rain"], w, color=BLUE,  alpha=0.8, label="Extreme rain months")
    ax6.set_xticks(x)
    ax6.set_xticklabels([f"{d}s" for d in extremes["decade"]], fontsize=8)
    ax6.set_title("Extreme events per decade (95th percentile)", fontsize=11, fontweight="bold", color=NAVY)
    ax6.set_ylabel("Count", fontsize=9)
    ax6.legend(fontsize=9)

    fig.suptitle(
        f"ERA5 Climate Trend Analysis — {region} (1980–2023)",
        fontsize=14, fontweight="bold", color=NAVY, y=0.98
    )
    fig.text(0.5, 0.01,
             "Data: ERA5 reanalysis (ECMWF / Copernicus CDS) | Analysis: Goutam Raman, Uni Hamburg",
             ha="center", fontsize=8, color="gray")

    out = "outputs/era5_climate_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Summary export
# ---------------------------------------------------------------------------

def export_summary(df: pd.DataFrame, trends: dict, region: str) -> None:
    df.drop(columns=["annual", "monsoon_df"], errors="ignore")
    summary = {
        "region": region,
        "period": f"{df['year'].min()}–{df['year'].max()}",
        "generated": datetime.now().isoformat(),
        "key_findings": {
            "warming_per_decade_C": trends["temperature"]["trend_per_decade"],
            "total_warming_C": trends["temperature"]["total_change"],
            "warming_significant": bool(trends["temperature"]["significant"]),
            "precip_trend_mm_per_decade": trends["precipitation"]["trend_per_decade_mm"],
            "monsoon_trend_mm_per_decade": trends["monsoon"]["trend_per_decade_mm"],
        }
    }
    with open("outputs/era5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    annual = trends["annual"]
    annual.to_csv("outputs/era5_annual_data.csv", index=False)
    print("  Exported → outputs/era5_summary.json")
    print("  Exported → outputs/era5_annual_data.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    region = "Kerala, India"
    print(f"\nERA5 Climate Analysis — {region}")
    print("=" * 50)

    print("  Generating ERA5-style dataset (1980–2023)...")
    df = generate_synthetic_era5(region=region)

    print("  Computing anomalies vs. 1981–2010 baseline...")
    df = compute_anomalies(df)

    print("  Running trend analysis...")
    trends = compute_trends(df)

    t = trends["temperature"]
    p = trends["precipitation"]
    m = trends["monsoon"]
    print(f"\n  Temperature trend : {t['trend_per_decade']:+.3f} °C/decade "
          f"(p={t['p_value']}, {'significant' if t['significant'] else 'not significant'})")
    print(f"  Total warming     : {t['total_change']:+.2f} °C over {df['year'].max()-df['year'].min()} years")
    print(f"  Precip trend      : {p['trend_per_decade_mm']:+.1f} mm/decade")
    print(f"  Monsoon trend     : {m['trend_per_decade_mm']:+.1f} mm/decade (JJAS)")

    print("\n  Generating plots...")
    plot_path = plot_full_report(df, trends, region)
    print(f"  Saved  → {plot_path}")

    export_summary(df, trends, region)
    print("\nDone.\n")
