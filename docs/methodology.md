# Methodology

## Research Objective

The project estimates last-meter delivery time at the building level for two
delivery modes:

- conventional car-based delivery;
- autonomous mobile robot (AMR) delivery.

The method combines geospatial features, Monte Carlo simulation, regression
models, and optional visual features extracted from Street View imagery.

## Baseline Pipeline

1. **Raw data ingestion**
   - NYC building footprints and attributes;
   - address points;
   - street centerlines;
   - neighborhood boundaries;
   - population data;
   - sidewalk data;
   - parking meter, parking regulation, and ParkNYC blockface data;
   - pedestrian mobility data.

2. **Geospatial feature engineering**
   - each building receives a delivery point;
   - each building is linked to the nearest street context;
   - neighborhood and population features are joined;
   - sidewalk, pedestrian, curbside, and land-use indicators are computed;
   - features are normalized for model training.

3. **Monte Carlo simulation**
   - car simulation models parking search outcome, walking distance, building
     entry, elevator time, and indoor delivery;
   - AMR simulation models robot approach, customer response, helper fallback,
     entry, elevator, and indoor delivery;
   - simulation outputs are mean and standard deviation of last-meter time.

4. **Regression modeling**
   - simulated buildings form the training sample;
   - candidate models are trained and evaluated;
   - the selected models are applied to the larger building dataset.

## Street View AI-Informed Pipeline

The visual branch adds building-level access observations:

- whether the image is usable;
- whether stairs are visible;
- whether gates or barriers are visible;
- whether ramps or ramp-like obstacles are visible;
- whether an AMR can plausibly reach the door.

These features are extracted from Street View imagery using an AI model. The
experiment was run on a subset of about 7,000 buildings due to API cost
constraints, but the workflow can be extended incrementally.

## Model Comparison

The AI-informed branch supports a clean comparison:

1. AI-informed simulated target times with only baseline geospatial features.
2. The same AI-informed simulated target times with visual AI features included.

This isolates whether the visual features add predictive value.

