"""Independent synthetic DST contract probe: real ingestion/adapters/cockpit, no cache IO."""
import json
from datetime import date, timedelta
import pandas as pd
from src.project_case import grid, emit_reserve_coopt, emit_da_id_reserve, AdapterUnavailableError
from src.dispatch import solve_joint_capacity_batch
from src.simulation import simulate_sequential_da_id_reserve_batch, simulate_da_id_reserve_ceiling_batch, simulate_stochastic_triple_batch
from src.time_utils import nominal_block_settlement_factors
from src.pages.simulation_cockpit import _reserve_coopt_total
from tests.test_project_case_adapters import _regelleistung_block_series, CB

results = {"source": "6 synthetic Regelleistung XLSX blocks at EUR 80/MW each, parsed to EUR 20/MW/h", "common": {"power_mw": 1, "availability": .95, "da": 0, "ida": 0}, "coopt": [], "triple": []}
for target in [date(2025,3,29),date(2025,3,30),date(2025,10,26),date(2026,3,29)]:
    idx = pd.DatetimeIndex(grid.expected_da_timestamps('DE_LU', target))
    da = pd.DataFrame({'price_eur_mwh': 0.0}, index=idx)
    reserve = _regelleistung_block_series(target)
    anc = pd.DataFrame({'capacity_price_eur_mw': reserve, 'product_type': 'FCR', 'zone':'DE_LU', 'direction':'symmetric'})
    common = dict(power_mw=1.0, duration_hours=1.0, efficiency=.88)
    pc = emit_reserve_coopt(da,reserve,zone='DE_LU',first_delivery_date=target,last_delivery_date=target,currency_basis=CB,reserve_product='FCR',reserve_source='synthetic_regelleistung',availability=.95,**common)
    cockpit = _reserve_coopt_total(da,'FCR',anc,valid_dates={target},tz='Europe/Berlin',**common)
    raw = solve_joint_capacity_batch(da,20.,tz='Europe/Berlin',**common)
    factors = nominal_block_settlement_factors(idx,timezone='Europe/Berlin',nominal_block_hours=4)
    results['coopt'].append({'date':str(target),'rows':len(idx),'physical_hours':float((pd.Timestamp(target+timedelta(days=1)).tz_localize('Europe/Berlin')-pd.Timestamp(target).tz_localize('Europe/Berlin'))/pd.Timedelta(hours=1)), 'nominal_hours':sum(h for _,h in grid.reserve_blocks('DE_LU',target)), 'factors': sorted(set(factors.tolist())), 'pc_cash':dict(pc.daily_realised_cash_series)[target], 'cockpit_cash':cockpit[0], 'joint_da_revenue':raw.iloc[0]['joint_da_revenue'], 'joint_capacity_revenue':raw.iloc[0]['joint_capacity_revenue'], 'profile':pc.adapter_provenance.expected_grid_profiles})
for target in [date(2025,3,30),date(2025,10,26),date(2026,3,28),date(2026,3,29)]:
    days=[target-timedelta(days=2), target-timedelta(days=1),target]
    da = pd.DataFrame({'price_eur_mwh':0.0},index=pd.DatetimeIndex([ts for d in days for ts in grid.expected_da_timestamps('DE_LU',d)]))
    ida = pd.DataFrame({'intraday_price_eur_mwh':0.0},index=pd.DatetimeIndex([ts for d in days for ts in grid.expected_ida_timestamps('DE_LU',d)]))
    reserve=pd.concat([_regelleistung_block_series(d) for d in days])
    row={'date':str(target)}
    try:
        pc=emit_da_id_reserve(da,ida,reserve,zone='DE_LU',first_delivery_date=target,last_delivery_date=target,currency_basis=CB,reserve_product='FCR',reserve_source='synthetic_regelleistung',bucket='hour_of_day',availability=.95,**common)
        row['pc_cash']=dict(pc.daily_realised_cash_series)[target]
    except AdapterUnavailableError as e: row['pc_unavailable']=str(e)
    seq,summary=simulate_sequential_da_id_reserve_batch(da,ida,reserve,dates=[target],tz='Europe/Berlin',bucket='hour_of_day',**common)
    row['cockpit_sequential_cash']=None if seq.empty else float(seq.iloc[0]['realised_eur'])
    row['cockpit_sequential_available']=summary['model_available']
    ceiling=simulate_da_id_reserve_ceiling_batch(da,ida,reserve,dates=[target],tz='Europe/Berlin',**common)
    row['cockpit_ceiling']=ceiling
    if target!=date(2025,3,30):
        stochastic,summ=simulate_stochastic_triple_batch(da,ida,reserve,dates=[target],tz='Europe/Berlin',bucket='hour_of_day',n_scenarios=1,seed=7,**common)
        row['cockpit_stochastic_cash']=None if stochastic.empty else float(stochastic.iloc[0]['stochastic_realised_eur'])
    results['triple'].append(row)
print(json.dumps(results,indent=2,default=str))
