# Current flooding and conditional future pressure warning

LED state = max(current_state, forecast_state). DANGER never directly means OPEN.
controller.py is unchanged.

| Signal | WARNING | DANGER |
| --- | --- | --- |
| Current outside depth | >=0.25m | >=0.30m |
| Current inside depth | >=0.25m | >=0.30m |
| Future equivalent depth crossing | T_critical 6..15 min | T_critical <=5 min |

## T_critical

Reuse RiskEngine._equivalent_depth: sqrt(abs(outside**2-inside**2)).
Hold the current indoor depth constant for all 15 external forecasts. Only
outside-higher pressure counts. The first equivalent depth >=0.30m defines
t_critical_min. A currently reached outside-higher threshold gives zero from valid paired
observations even without a READY forecast or known/inactive equalization.
pressure_forecast_status is CURRENT_CRITICAL_REACHED; forecast_status is preserved. Analytical boundaries include floating point tolerance.

Both depths use the same door-bottom datum. The existing calculation assumes
still water, equal density and both surfaces below the door top. T_critical is
a conditional estimate, not an experimentally verified door-opening deadline.

Missing indoor depth or numeric overflow makes pressure time unavailable.
Unless the current threshold is already reached, invalid/unavailable forecasts
or active/unknown equalization make future risk WARNING and its time unavailable. Known
current DANGER always wins. None can mean no crossing or unavailable: inspect
pressure_forecast_status.

Current equivalent depth, signed pressure and pressure direction remain diagnostic.
There is no current equivalent-depth WARNING/DANGER LED rule. Equalized .50/.50m
still gives flooding DANGER. Reverse pressure is recorded but not counted as an
outside-higher future crossing.

## Result contract

- t_critical_min: conditional equivalent-depth .30m time used for future LED risk.
- t_external_warning_min: configured external forecast-threshold time, diagnostic only.
- t_external_critical_min: external .30m time, diagnostic only.
- current_state / forecast_state: independent current/future severity.
- current_external_depth_m / current_internal_depth_m / current_equivalent_depth_m:
  descriptive property accessors.

Legacy constructor fields (including t_warning_min) remain for controller/caller
compatibility. t_pressure_critical_if_internal_constant_min mirrors t_critical_min.
Properties are not included by dataclasses.asdict; explicitly serialize new names.

RiskEngine's default external forecast threshold is 0.30m only as an
uncalibrated mock/offline fallback. A live session refuses a CV checkpoint or
any checkpoint without a locked policy. Final training stores the reviewed
external warning threshold in the checkpoint; live runtime constructs
RiskEngine with that same value. The separate pressure equivalent-depth
critical threshold remains 0.30m.

Previous external-threshold event scores do not measure this new LED policy.
Evaluate it separately with paired indoor/outdoor scenarios. quick/accumulate
remain mock tests; this change does not add a live LSTM runner. command stays NONE.
