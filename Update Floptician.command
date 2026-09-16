#!/bin/bash
cd "$(dirname "$0")" || exit 1
/bin/bash scripts/mac/manage.sh update
result=$?
read -r -p "Press Enter to close..."
exit "$result"
