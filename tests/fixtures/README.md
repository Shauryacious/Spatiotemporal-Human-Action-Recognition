# Fixtures

`uv run python scripts/make_fixtures.py` creates original FFmpeg test-pattern videos:
4 seconds at 12 FPS, variable frame spacing, and a 90-degree display rotation.
There is no third-party media license or real action ground truth involved. Generated
binaries stay outside Git. Tests generate isolated copies in pytest temporary directories.
`model_responses.json` contains provider-neutral validation cases.
