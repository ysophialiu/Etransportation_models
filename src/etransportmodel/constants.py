import numpy as np

SHAPEFILE = "example_data/Model_Traffic_Analysis_Zones_2020.shp"
TRIPFILE = "example_data/EV_trip.p"
EN = {0:60, 1:100, 2:100}
CN = {0:0.3, 1:0.3, 2:0.35}
PROB_EN = [0.3,0.6,0.1]
SNR = [0,1,2]
CHARGE_BEHAVE = 'dislike_fast_charge'
RATE = np.array([3.6,6.2,150])
RATE_NAME =  ['h2','l2','l3']
LOCATION_NAME = ['home','work','public']
HOME_PRICE = 0.13
D = 2
TEST_PUB_PRICE = 0.43
L_AVAILABLE = [0,1,1]
PUB_PRICE = 0.43
ZZONES = 'SD10'
YEAR = 10
DISCOUNT_RATE = 0.05
SCALE_TO_YEAR = 365
STATION_EFFICIENCY = 0.95
ELECTRICTIY_COST = 0.11
L2_BUY_COST = 3450
L2_BUILD_COST = 3000
DCFC_BUY_COST = 25000
DCFC_BUILD_COST = 21000