#!/bin/bash
# CPU-only checks (no GPU, no model download): engine unit tests + the data/eval/cf/report stages with a fake model.
set -e; cd "$(dirname "$0")/.."
python tests/test_engine.py
python tests/test_cpu_pipeline.py > /tmp/g2t_cpu_test.log 2>&1 && echo "CPU pipeline test passed (log: /tmp/g2t_cpu_test.log)"
