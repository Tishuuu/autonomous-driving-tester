@echo off
setlocal EnableDelayedExpansion

for %%%%I in (
1778082371012
1778082702309
1778082985541
1778083253695
1778083509606
1778083832739
1778084432314
1778084874559
1778085263718
1778085427098
1778085692205
1778085862895
1778086053555
) do (
  set "BASE=TEST_%%%%I"
  set "SOURCEDIR="
  set "REBUILDDIR=analysis_exports\TEST_%%%%I_FULL_REBUILD"
  set "M16DIR=analysis_exports\TEST_%%%%I_M16_FULL"

  for /d %%%%D in (analysis_exports\TEST_%%%%I*) do (
    if not defined SOURCEDIR (
      if exist "%%%%D\input_video.mp4" (
        if exist "%%%%D\raw_sensors.json" set "SOURCEDIR=%%%%D"
        if exist "%%%%D\sensors.json" set "SOURCEDIR=%%%%D"
        if exist "%%%%D\1_raw_sensors.csv" set "SOURCEDIR=%%%%D"
      )
    )
  )

  if not defined SOURCEDIR (
    echo SKIP !BASE! - missing input_video.mp4 or sensors file
  ) else (
    set "VIDEO=!SOURCEDIR!\input_video.mp4"

    if exist "!SOURCEDIR!\raw_sensors.json" (
      set "SENSORS=!SOURCEDIR!\raw_sensors.json"
    ) else if exist "!SOURCEDIR!\sensors.json" (
      set "SENSORS=!SOURCEDIR!\sensors.json"
    ) else (
      set "SENSORS=!SOURCEDIR!\1_raw_sensors.csv"
    )

    echo.
    echo ================ !BASE! FULL BACKEND ================
    echo Source: !SOURCEDIR!
    echo Video: !VIDEO!
    echo Sensors: !SENSORS!

    python scripts\replay_pipeline.py --test-id "!BASE!_FULL_REBUILD" --video "!VIDEO!" --sensors "!SENSORS!" --output-dir "!REBUILDDIR!"

    if not exist "!REBUILDDIR!\2_feature_vector.csv" (
      echo FAILED !BASE! - no 2_feature_vector.csv created
    ) else (
      echo.
      echo ================ !BASE! M16 SCORING ================
      python scripts\replay_pipeline_m16.py --test-id "!BASE!_M16_FULL" --feature-vector "!REBUILDDIR!\2_feature_vector.csv" --output-dir "!M16DIR!"
    )
  )
)
