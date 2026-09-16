#!/bin/bash
cd "$(dirname "$0")" || exit 1
/bin/bash scripts/mac/manage.sh run
result=$?
if (( result != 0 )); then read -r -p "Press Enter to close..."; fi
exit "$result"
