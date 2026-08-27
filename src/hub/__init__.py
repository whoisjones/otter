# Sources for the model repositories published to the Hugging Face Hub.
#
# `configuration_otter.py` and `modeling_otter.py` are the `trust_remote_code`
# modules uploaded verbatim to each model repository. They are templates, not
# modules of this package: they import `masks`, `loss`, `metrics` and
# `collate_fn` as siblings, which only exist next to them once
# `publish_to_hub.py` has generated those from `src/` into a staging directory.
# Importing them from inside this repository will therefore fail, by design --
# it is what keeps the published copies from drifting away from `src/`.
