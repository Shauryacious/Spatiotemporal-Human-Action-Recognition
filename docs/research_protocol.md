# Research protocol

Freeze model ID, resolved model version, prompt hash, query, configuration, alias dictionary,
data hashes, split groups, seed, sampling policy, and budget before held-out evaluation.
Use train/development examples for prompt design, validation for tuning, and test once frozen.
Check media hash, original source, session group, and subject IDs for cross-split leakage.
Hosted-model pretraining exposure to public benchmarks cannot be excluded.

B1 is recognition only; temporal metrics are not applicable. B2 adds temporal proposals and
boundary evidence. B3 adds one spatial pass. B4 adds bounded alternating refinement. Compare
with equal maximum budgets and report actual calls, known tokens/cost, partial coverage,
failures, cache warmth, and latency. Use three repeats when the provider budget permits.
The mock pipeline validates engineering only and establishes no recognition accuracy.

Temporal AP uses class-wise stable score ordering, highest-IoU unmatched GT matching,
one match per GT, a monotone precision envelope, and integration at recall changes.
Unknown scores receive evaluation rank 0.0. Classes absent from ground truth are excluded
from mAP and listed explicitly with their false positives. Aliases must be frozen on
validation data. The diagnostic [0.5,0.75,0.95] sweep is not an official THUMOS protocol.
Use each benchmark's official evaluator separately when making benchmark comparisons.

Spatial/joint evaluation uses only GT-visible observed boxes and a frozen timestamp
matching tolerance of 0.125 seconds. Missing observations count as zero. One consistent
predicted track is assigned to each GT participant for the event, never switched per frame.
Ambiguous tracks cannot satisfy joint localization. Multi-participant assignment is one-to-one.
No spatial/joint accuracy claim is possible without reviewed box annotations.

Before research release, obtain licensed/consented pilot and held-out videos, reviewed
annotations, and an accessible model/quota profile. Report negative refinement results,
missed short actions, failures, and exclusions. Do not infer causation or intent from action labels.
