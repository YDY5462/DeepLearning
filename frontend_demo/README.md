# Frontend Demo (Real Output Version)

## 1) Build real data bundle

Run from repo root:

```bash
python frontend_demo/build_model_bundle.py
```

## 1.1) Build real station names (optional but recommended)

Run:

```bash
python frontend_demo/build_station_name_map.py
```

Then edit:

- `frontend_demo/station_name_map.csv`

Columns:

- `station_id` (`S001` ... `S276`)
- `station_name` (real station name)
- `line_name` (optional line label)

After editing, run the same command again to regenerate:

- `frontend_demo/station_name_map.js`

This command reads:

- `final result/15/predictions.csv`
- `final result/15/Y_test_original.csv`
- `final result/15/*_merge.txt`
- `saved_runs/baseline_15min_provenance.csv`

and generates:

- `frontend_demo/model_bundle.js`

## 2) Open pages

- Main page: `frontend_demo/index.html`
- Station page: `frontend_demo/station.html`

## 3) What this demo shows

- No random mock data.
- All charts are driven by real test-set matrix values.
- Supports sample slider + autoplay for smooth presentation.
- Shows true value, prediction, absolute error, APE, and source/provenance info.

## 4) Key files

- `build_model_bundle.py`: build script for frontend data bundle.
- `model_bundle.js`: generated real data package.
- `demo_data.js`: shared state + data selectors.
- `map_app.js`: dashboard logic.
- `station_app.js`: station detail logic.
- `styles.css`: unified visual style.
