#!/bin/bash
# per-file test loop; a finished file is recorded in output/tests_done.txt and skipped on the next start
cd "$(dirname "$0")/.."
touch output/tests_done.txt
for f in tests/test_*.py; do
  grep -qx "$f" output/tests_done.txt && continue
  python -m pytest "$f" -q -p no:cacheprovider > "output/t_$(basename $f).log" 2>&1
  echo "$f rc=$?" >> output/tests_summary.txt
  echo "$f" >> output/tests_done.txt
done
echo ALLDONE >> output/tests_summary.txt
