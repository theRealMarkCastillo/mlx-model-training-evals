# Historical artifacts

The original top-level JSON reports, chart, and adapter config in this directory predate the corrected dataset and evaluator. They are preserved for historical inspection, including local modifications, and must not be cited as results of the corrected pipeline.

Regenerate the dataset and retrain before measuring current quality. New runs write to unique `runs/<kind>-<id>/` directories and publish a `latest_<kind>.json` manifest only after successful completion. The training runner publishes an adapter pointer at the configured `adapter_path/latest.json`. It does not overwrite the old top-level artifacts.
