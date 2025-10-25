#!/bin/bash

cd /path/to/your/project  # change this to your project folder
git add .
git commit -m "Auto update $(date '+%Y-%m-%d %H:%M:%S')"
git push origin main
