
# author: steeve laquitaine
# purpose: setup and build project
#   
# usage:
# 
#   ```bash
#   bash setup.sh
#   # Requirements built! Please update requirements.in if you'd like to make a 
#   # change in your project's dependencies, and re-run build-reqs to generate 
#   # the new requirements.txt.
#   ````

kedro build-reqs
kedro install