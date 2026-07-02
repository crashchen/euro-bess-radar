# Validate the reBAP / imbalance overlay with Netztransparenz CSVs

This runbook validates the passive imbalance-settlement overlay with real
German Netztransparenz data, either through the live fetch button or through
manual CSV exports converted to the unified import template.

## Live fetch path

The app can fetch German DE_LU reBAP/NRV data directly from the public
Netztransparenz chart CSV handler.

1. Run the app:

   ```bash
   streamlit run app.py
   ```

2. Select `DE_LU` and a date window, for example May 2026.
3. Sidebar → **Ancillary Services Data** → **Unified reBAP / Imbalance CSV**.
4. Click **Fetch Netztransparenz reBAP/imbalance**.
5. Data Trust should show `Netztransparenz.de` in the
   `reBAP / imbalance settlement sources` table.

Use the manual path below when you need to reproduce a provider file exactly or
when the public download handler is unavailable.

## Inputs

Download the timestamped CSV exports from Netztransparenz:

- `NRV-Saldo qualitaetsgesichert [...].csv`
- `reBAP unterdeckt [...].csv` or `reBAP ueberdeckt [...].csv`

The reBAP export used by the converter must contain both `reBAP unterdeckt` and
`reBAP ueberdeckt`. For the symmetric German reBAP series those columns should
be equal; the converter rejects the file if they differ.

Ignore `reBAP Punktewolke` and `reBAP Histogramm` exports for the model path:
they are chart/statistics data and do not preserve the full 15-minute time axis.

## Convert to the unified import template

```bash
python scripts/convert_netztransparenz_imbalance.py \
  --nrv "Netztransparenz/NRV-Saldo qualitaetsgesichert [2026-07-02 00-46-52].csv" \
  --rebap "Netztransparenz/reBAP unterdeckt [2026-07-02 00-43-52].csv" \
  --out samples/netztransparenz_imbalance_de_lu_2026-05.csv
```

Expected output for the May 2026 sample:

```text
Wrote 2976 unified imbalance rows ...
Sign convention: positive system_imbalance_volume_mw = system short; negative = system long.
```

The output schema is the standard app upload format:

```csv
timestamp,zone,imbalance_price_eur_mwh,system_imbalance_volume_mw
2026-04-30T22:00:00Z,DE_LU,122.73,391.596
```

## What the converter validates

- German local `Datum` + `von` + `Zeitzone` timestamps convert cleanly to UTC.
- NRV-Saldo and reBAP are regular 15-minute time series.
- NRV-Saldo and reBAP timestamps align exactly.
- `reBAP unterdeckt` and `reBAP ueberdeckt` are equal for the symmetric German
  reBAP export.
- The generated CSV is accepted by `parse_imbalance_import_csv`.

## Upload and inspect the converted file

1. Run the app:

   ```bash
   streamlit run app.py
   ```

2. Sidebar → **Ancillary Services Data** → **Unified reBAP / Imbalance CSV**.
3. Upload `samples/netztransparenz_imbalance_de_lu_2026-05.csv`.
4. Data Trust tab:
   - Coverage matrix shows `imbalance_settlement` for `DE_LU`.
   - `reBAP / imbalance settlement sources` shows a `Manual CSV` row (or
     `Mixed` if the same zone also contains live-fetched intervals).
5. Simulation Cockpit:
   - Select a DE_LU date window overlapping May 2026.
   - Open **reBAP / imbalance overlay (historical replay)**.
   - Confirm the panel shows a signed overlay metric plus `system_short` /
     `system_long` breakdown.

## Interpretation red lines

- Positive `system_imbalance_volume_mw` means system short / undercovered;
  positive BESS net dispatch helps.
- Negative `system_imbalance_volume_mw` means system long / overcovered;
  negative BESS net dispatch helps.
- `imbalance_price_eur_mwh` is kept as the signed published cash-flow price.
  Do not apply an additional direction sign flip.
- The overlay is not additive to DA/IDA/reserve/activation strategy totals. It
  ignores SoC and energy sustainability, so treat it as a historical passive
  settlement diagnostic, not a live BRP-control model.
