# Automated reproduction checks

The 2026-10-02 local acceptance run passed 117 tests and reproduced all 360
original benchmark cases through `scripts/reproduce.py`. The independent
numerical verifier passed, and all seven numeric reports matched the published
baseline with zero numerical difference in this environment.

- [Local acceptance and execution-script hashes](local_acceptance.json)
- [Independent numerical reconstruction](numerical_verification.json)
- [All seven reports compared against the baseline](numerical_regression.json)
- [Cross-platform checks](https://github.com/f0rKyrie1rving/music-probability-calibration/actions/workflows/reproduce.yml)
- [Known platform differences and retained failures](PORTABILITY.md)
- [Five-draw stability checks](https://github.com/f0rKyrie1rving/music-probability-calibration/actions/workflows/stability.yml)

Recovery tests inject interruption into the real case-execution loop and check
that the original partial files are archived, completed cases remain present,
and no case is omitted. They also exercise dead process recovery, live writer
rejection, report protection and source/data/environment integrity checks.
Unknown or corrupt state is preserved and stops execution; it is not silently
classified as an ordinary interruption.

These checks establish computational consistency. They do not show better
performance on new music, and they do not establish that the source tags are
complete or human-confirmed.
