# Change Log
All notable changes to this project will be documented in this file.

## [0.1.2] - 2023-11-07

This update is to fix a crucial but small bug. 

### Added
- The documentation has been extended (though far from finalized).

### Changed

### Fixed
- The `__init__.py` files in the `anomaly_detection` module were updated
  to properly import classes that are not directly in the `anomaly_detection`,
  but rather in a sub folder. 

## [0.1.1] - 2023-10-26

This update doesn't include a lot of changes. It only slightly modified the
readme.

### Added
- Added an official release to the repository, and a badge to indicate
  the latest release. 

### Changed

### Fixed
- Fixed the link to the image showcasing the anomaly scores of an 
  IForest on a Demo time series. 

## [0.1.0] - 2023-10-26

First release of `dtaianomaly`! While our toolbox is still a work in progress, 
we believe it is already in a usable stage. Additionally, by publicly releasing 
`dtaianomaly`, we hope to receive feedback from the community! Be sure to check 
out the [documentation](https://u0143709.pages.gitlab.kuleuven.be/dtaianomaly/)
for additional information!

### Added
- `anomaly_detection`: a module for time series anomaly detection algorithms. 
   Currently, basic algorithms using[PyOD](https://github.com/yzhao062/pyod) 
   are included, but we plan to extend on this in the future!

- `data_management`: a module to easily handle datasets. You can filter the datasets on 
   certain properties and add new datasets through a few simple function calls! More 
   information can be found in the [Documentation](https://u0143709.pages.gitlab.kuleuven.be/dtaianomaly/getting_started/data_management.html). 

- `evaluation`: It is crucial to evaluate an anomaly detector in order to quantify its 
   performance. This module offers several metrics to this end. `dtaianomaly` offers 
   traditional metrics such as precision, recall, and F1-score, but also more recent 
   metrics that were tailored for time series anomaly detection such as the
   [Affiliation Score](https://dl.acm.org/doi/10.1145/3534678.3539339)
   and [Volume under the surface (VUS)](https://dl.acm.org/doi/10.14778/3551793.3551830)

- `visualization`: This module allows to easily visualize the data and anomalies, as 
   time series and anomalies inherently are great for visual inspection.

- `workflow`: This module allows to benchmark an algorithm on a larger set of datasets, 
   through configuration files. This methodology ensures reproducibility by simply providing 
   the configuration files! 
 
### Changed
 
### Fixed

