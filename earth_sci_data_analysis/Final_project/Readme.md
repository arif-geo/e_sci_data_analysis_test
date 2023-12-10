This final project sub-repository contains the codes and data for my final project.
* Whole project is divided into several tasks:
* Part 1: Selection of earthquakes that follow some conditions using pandas
* Part 2: Downloading earthquake `mseed` (miniseed) files using `Obspy` package
* Part 3: `(not included here)` Picking earthquake arrival times at each station using `PyRocko` package which is a UI based software and the `mseed` files from `Part 2`. This will produce a `txt` file containing earthquake arrival times, P-wave polarity (positve/negative), station location for each event ID. This is a manual process.
* Part 4: Earthquake location determination using `grid search` method and checking location accuracy of depth given by USGS. This uses a lot of computing power as the whole study area will be divided into 3D cubes and calculate seismic raypath from source to station. I will try to use `parallel computing` in this step.
* Part 5: Determine `Focal mechanism` of each major earthquake using `HASH` software (a fortran program widely used for this purpose). Focal mechanism is displayed as a `beachball` which shows areas of compressional and dialational stress around an earthquake.
* Part 6: Use machine learning algorithms to determine `Focal mechanism` and compare them with calculated results from the `HASH` program.