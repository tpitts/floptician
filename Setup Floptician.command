#!/bin/bash
cd "$(dirname "$0")" || exit 1
echo "1. Install/check setup (default)"
echo "2. Change OBS password"
read -r -p "Choose 1 or 2: " choice
if [[ "$choice" == 2 ]]; then
    /bin/bash scripts/mac/manage.sh password
else
    /bin/bash scripts/mac/manage.sh setup
fi
result=$?
read -r -p "Press Enter to close..."
exit "$result"
