# Parking Availability Treatment

Parking availability is treated as a structural curbside difficulty proxy, not
as real-time parking occupancy.

## Raw Sources

- active on-street parking meters;
- parking regulation signs;
- ParkNYC blockface information;
- parking rate zones.

## Street-Level Aggregation

Parking meters and curb restriction signs are spatially joined to street
centerlines using a 20 meter buffer. For each street segment, the pipeline
computes:

- active parking meter count;
- restrictive sign count;
- nearest blockface vehicle type;
- parking lane count, when available.

## Final Curb Difficulty Score

The final car-side parking feature is:

```text
curb_parking_difficulty_penalty =
  (
    parking_meter_scarcity_penalty
    + curb_restriction_intensity_norm
    + commercial_or_truck_curb_context_flag
  ) / 3
```

Interpretation:

- fewer active meters means higher scarcity;
- more restrictive signs means higher restriction intensity;
- commercial/truck context means higher curb competition.

## Use in Simulation

The car Monte Carlo simulation estimates a geometric parking capacity from the
nearest street segment length. The curb difficulty score is then used as the
simulated occupancy ratio:

```text
occupied_spaces = parking_capacity * curb_parking_difficulty_penalty
```

For each run, occupied spots are sampled randomly and the driver chooses the
nearest available spot to the delivery point.

## Limitation

This is not real-time parking availability. It does not model time of day,
day of week, actual turnover, or dynamic parking demand. It is a reproducible
citywide proxy for curbside parking difficulty.

