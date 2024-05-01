# etransportmodels

Transportation electrification, which refers to replacing fossil fuel use in the transportation sector with electricity, has evolved into a global mission to reduce greenhouse gas (GHG) emissions from the transportation sector. As an action, promoting the adoption of battery electric vehicles (BEVs), vehicles that are powered purely by electricity and produce zero emissions when in use, has emerged as a promising solution to an electrified low-emission transportation system.

Despite widespread enthusiasm for BEVs, many research questions need to be answered in order to achieve large-scale BEV adoption and emissions goals, such as how to quantify the spatio- temporal changes in charging demand, how to build charging infrastructure in an economically sustainable way, and how to reduce emissions by coordinating renewable energy and charging planning.

Our research took a system modeling approach and proposed several mathematical models and optimization algorithms to help answer these questions. This package provides several main modules and a notebook to run everything together:

**Link to website:** https://etransport.cee.cornell.edu/

## Charging Choice module
This module provides functions to calculate the indirect utilities of the charging choices made relating to charging mode, state of charge preferences, and cost of charging.

## Charging Demand module
This module provides an tripdata-based simulation model and delivers BEV charging demand estimation.

## Charging Placement module
This module presents a charging station placement model based on activity-based simulation that finds the optimized location and capacity of multi-type charging stations based on their net present values.

## Optimization Solver module
This module contains all the solution algortihms to select Electric Vechile Charging Infastructure placements and updates the best-observed results.

## Example Notebook
This notebook contains all the functions and modules used in the charging choice, charging demand, charging placement, and optimization solver modules. It allows users to run the models and customize their case studies on their local computers.

## How to install
Clone the repository or download the zip file onto your local machine.

```sh
git clone https://github.com/ysophialiu/Etransportation_models.git
```

Change directory to be inside the cloned repository/downloaded folder and install it..
```
cd ~/Etransportation_models
pip install .
```


