import geopandas as gpd
import pickle
import numpy as np
import torch
import pandas as pd
from scipy.stats import truncnorm

from etransportmodel.constants import *


class TripData():
    """A class to store all information related to the current trip information."""

    def __init__(
            self,
            shapefile: str = SHAPEFILE, 
            tripfile: str = TRIPFILE,
            en: dict = None,
            cn: dict = None,
            prob_en: list = None,
            snr: list = None,
            charge_behave: str = CHARGE_BEHAVE,
            rate : list = None,
            rate_name: list  = None,
            location_name: list = None,
            home_price: float = HOME_PRICE,
            simulated_days: int = SIMULATED_DAYS,
            L_available: list = None,
            pub_price: float = PUB_PRICE,
            zzones: str = ZZONES,
            year: int = YEAR, 
            discount_rate: float = DISCOUNT_RATE,
            scale_to_year: int = SCALE_TO_YEAR,
            station_efficiency: float = STATION_EFFICIENCY,
            electricity_cost: float= ELECTRICTIY_COST, 
            L2_buy_cost: float = L2_BUY_COST,
            L2_build_cost: float = L2_BUILD_COST, 
            DCFC_buy_cost: float = DCFC_BUY_COST, 
            DCFC_build_cost: float = DCFC_BUILD_COST,
        ):
        """Creates an instance of TripData with all attributes related to trip data, charging choice,
        charging demand, and charging placement. 

        Args:
            shapefile (str, optional): _description_. Defaults to SHAPEFILE.
            tripfile (str, optional): _description_. Defaults to TRIPFILE.
            en (dict, optional): _description_. Defaults to None.
            cn (dict, optional): _description_. Defaults to None.
            prob_en (list, optional): _description_. Defaults to None.
            snr (list, optional): _description_. Defaults to None.
            charge_behave (str, optional): _description_. Defaults to CHARGE_BEHAVE.
            rate (list, optional): _description_. Defaults to None.
            rate_name (list, optional): _description_. Defaults to None.
            location_name (list, optional): _description_. Defaults to None.
            home_price (float, optional): _description_. Defaults to HOME_PRICE.
            simulated_days (int, optional): _description_. Defaults to SIMULATED_DAYS.
            L_available (list, optional): _description_. Defaults to None.
            pub_price (float, optional): _description_. Defaults to PUB_PRICE.
            zzones (str, optional): _description_. Defaults to ZZONES.
            year (int, optional): _description_. Defaults to YEAR.
            discount_rate (float, optional): _description_. Defaults to DISCOUNT_RATE.
            scale_to_year (int, optional): _description_. Defaults to SCALE_TO_YEAR.
            station_efficiency (float, optional): _description_. Defaults to STATION_EFFICIENCY.
            electricity_cost (float, optional): _description_. Defaults to ELECTRICTIY_COST.
            L2_buy_cost (float, optional): _description_. Defaults to L2_BUY_COST.
            L2_build_cost (float, optional): _description_. Defaults to L2_BUILD_COST.
            DCFC_buy_cost (float, optional): _description_. Defaults to DCFC_BUY_COST.
            DCFC_build_cost (float, optional): _description_. Defaults to DCFC_BUILD_COST.
        """        
        self.initialize_trip_data(shapefile, tripfile, en, cn, prob_en, snr, charge_behave)
        self.initialize_charging_choice(rate, rate_name, location_name, home_price, simulated_days, L_available)
        self.intiialize_charging_demand(pub_price, zzones)
        self.initialize_charging_placement(year, discount_rate, scale_to_year, station_efficiency, electricity_cost, 
                                           L2_buy_cost, L2_build_cost, DCFC_buy_cost, DCFC_build_cost)


    def initialize_trip_data(self, shapefile: str, tripfile: str, en: dict, cn: dict, 
                             prob_en: list, snr: list, charge_behave: str):
        """Initializes TripData module with information related to the shapefile and tripfile.

        Args:
            shapefile (str): shapefile path
            tripfile (str): tripfile path
            en (dict): EV battery capacity in kWh
            cn (dict): energy consumption rate in kWh/mile
            prob_en (list): probability distribution of each EV battery capacity type
            snr (list): keys of the distribution types
            charge_behave (str): charging behavior settings
        """

        self.shapefile = gpd.read_file(shapefile)
        self.ev_trip = pickle.load(open(tripfile, "rb")) 
        self.ev_sample = self.ev_trip["EV_list"].unique()
        self.ev_n_trips = self.ev_trip['EV_list'].value_counts()

        if en is None:
            en = EN
        self.en = en

        if cn is None:
            cn = CN
        self.cn = cn

        if prob_en is None:
            prob_en = PROB_EN
        self.prob_en = prob_en

        if snr is None:
            snr = SNR
        self.snr = snr

        self.charge_behave = charge_behave

        self.Scenario = np.random.choice(self.snr, len(self.ev_sample), p=self.prob_en)
        self.en_v = [self.en.get(n) for n in self.Scenario]
        self.cn_v = [self.cn.get(n) for n in self.Scenario]
        self.en_v = dict(zip(self.ev_sample, self.en_v))
        self.cn_v = dict(zip(self.ev_sample, self.cn_v))

        self.mean, self.sd, self.low, self.upp = self.Init_SOC(self.charge_behave)
        self.SOC_int_ = self.get_truncated_normal(self.mean, self.sd, self.low, self.upp).rvs(size=len(self.ev_sample))
        self.SOC_int = dict(zip(self.ev_sample, self.SOC_int_))

        
    def initialize_charging_choice(self, rate: list, rate_name: list, location_name: list, home_price: float, simulated_days: int, L_available: list) -> None:
        """Initializes TripData module with information related to charging choice.

        Args:
            rate (list): charging rate in kWh
            rate_name (list): charger type 
            location_name (list): location code name
            home_price (float): home charging price in $/kWh
            simulated_days (int): number of days to simulate, D > 1
            L_available (list): list of chargers available, 0 for not available, 1 for available e.g. [0, 1, 1]
        """

        if rate is None:
            rate = RATE
        self.rate = rate

        if rate_name is None:
            rate_name = RATE_NAME
        self.rate_name = rate_name

        if location_name is None:
            location_name = LOCATION_NAME
        self.location_name = location_name

        self.home_price = home_price

        self.simulated_days = simulated_days

        if L_available is None:
            L_available = L_AVAILABLE
        self.L_available = L_available

        self.charging_behavior_parameter(self.charge_behave)
    

    def intiialize_charging_demand(self, pub_price:float, zzones: str) -> None:
        """Initialize TripData module with information related to charging demand.

        Args:
            pub_price (float): public charging price in $kWh
            zzones (str): zone name, ID from geo data
        """

        self.pub_price = pub_price
        self.zzones = zzones
        self.num_zone = len(self.shapefile[self.zzones].unique())

        zone_array = np.sort(self.shapefile[self.zzones].unique()).tolist()
        self.shapefile['new_zone_name'] = [zone_array.index(x)+1 for x in self.shapefile[self.zzones]]

        join_zones = pd.DataFrame(self.shapefile[['OBJECTID','new_zone_name']]).set_index('OBJECTID')
        self.ev_trip = self.ev_trip.join(join_zones, on="d_taz")
        self.ev_trip['d_taz']=self.ev_trip['new_zone_name']
        self.ev_trip = self.ev_trip.drop(columns =['new_zone_name'])


    def initialize_charging_placement(self, year: int, discount_rate: float, scale_to_year: int, station_efficiency: float, electricity_cost: 0.11, 
                                      L2_buy_cost: float, L2_build_cost: float, DCFC_buy_cost: float, DCFC_build_cost: float) -> None:
        """ Initializes TripData module with information realted to charging placement.

        Args:
            year (int): life cycle times in years
            discount_rate (float): discount rate over time
            scale_to_year (int): number of days in particular year
            station_efficiency (float): efficiency of charging station
            electricity_cost (float): electricity cost in $/kWh: U.S average 0.07 for industry; 0.11 for commercial
            L2_buy_cost (float): L2 Charger Buy Cost in $
            L2_build_cost (float): L2 Charger Build Cost in $
            DCFC_buy_cost (float): DCFC Charger Buy Cost in $
            DCFC_build_cost (float): DCFC Charger Build Cost in $
        """

        self.ddtype = torch.double
        self.year = year
        self.discount_rate = discount_rate
        self.scale_to_year = scale_to_year
        self.station_efficiency = station_efficiency
        self.electricity_cost = electricity_cost
        self.L2_buy_cost = L2_buy_cost
        self.L2_build_cost = L2_build_cost
        self.DCFC_buy_cost = DCFC_buy_cost
        self.DCFC_build_cost = DCFC_build_cost

        self.present_worth_factor = ((1 + discount_rate) ** year -1)/(discount_rate * (1 + discount_rate) ** year)
        
        self.invest_cost = [self.L2_buy_cost + self.L2_build_cost] * self.num_zone
        self.invest_cost.extend([DCFC_buy_cost + DCFC_build_cost] * self.num_zone)
        self.invest_cost = torch.tensor(self.invest_cost,dtype = self.ddtype)


    def Init_SOC(self, cases: str) -> list:
        """Define EV initial state of charge distribution.

        Args:
            cases (str): charging behavior

        Returns:
            list: mean, sd, low, and upp of charging behavior
        """
        parameter_list = [[0.7324919595528574, 0.14222975174228758, 0.22067026978914075, 1.0],
                        [0.5734890448698912, 0.15712310015212297, 0.20243999999999995, 1.0],
                        [0.8276271161103358, 0.12204207789138573, 0.2041829411620677, 1.0],
                        [0.7477273429502866, 0.14006812858346473, 0.20587000000000003, 1.0],
                        [0.7207728408826842, 0.14543499457298006, 0.20671258988867014, 1.0],
                        [0.7029625189454968, 0.15211267134712808, 0.11100674272163669, 1.0],
                        [0.6297560710721821, 0.17206166873501583, 0.18099748205730337, 1.0]]
                        
        
        if cases == 'base':
            res = parameter_list[0]
        if cases == 'low_risk_sensitive':
            res = parameter_list[1]
        if cases == 'high_risk_sensitive':
            res = parameter_list[2]
        if cases == 'prefer_fast_charge':
            res = parameter_list[3]
        if cases == 'dislike_fast_charge':
            res = parameter_list[4]
        if cases == 'high_cost_sensitive':
            res = parameter_list[5]
        if cases == 'low_range_buffer':
            res = parameter_list[6]
        return res


    def get_truncated_normal(self, mean: float, sd: float, low: float, upp: float) -> int:
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    

    def charging_behavior_parameter(self, cases: str) -> None:
        """Initializes charging behavior parameters: 
        beta_SOC, beta_R, beta_delta_SOC, beta_0, beta_cost, beta_SOC_0, lbd
        [beta_SOC] represents the level of risk sensitivity, the larger the value, the more user is disposed to charging.
        [beta_R] represents the marginal utility of the charging rate
        [beta_delta_SOC] represents the rate of change for the level of risk sensitivity
        [beta_0] represents a constant of 1
        [beta_cost] represents the marginal utility of cost
        [beta_SOC_0] fixed SOC parameter
        [lbd] error Distribution

        Args:
            cases (str): charging behavior
        """        
        parameter_list = [[3,0,2,1,0.1,0.3,1],
                        [8,0,2,1,0.1,0.3,1],
                        [2,0,2,1,0.1,0.3,1],
                        [3,0.005,2,1,0.1,0.3,1],
                        [3,-0.005,2,1,0.1,0.3,1],
                        [3,0,2,1,0.2,0.3,1],
                        [3,0,2,1,0.1,0.2,1]]
                        
        if cases == 'base':
            res = parameter_list[0]
        if cases == 'low_risk_sensitive':
            res = parameter_list[1]
        if cases == 'high_risk_sensitive':
            res = parameter_list[2]
        if cases == 'prefer_fast_charge':
            res = parameter_list[3]
        if cases == 'dislike_fast_charge':
            res = parameter_list[4]
        if cases == 'high_cost_sensitive':
            res = parameter_list[5]
        if cases == 'low_range_buffer':
            res = parameter_list[6]
        
        (
            self.beta_SOC,
            self.beta_R,
            self.beta_delta_SOC,
            self.beta_0,
            self.beta_cost,
            self.beta_SOC_0,
            self.lbd
        ) = res


    def summary(self):
        """Prints a summary of all parameter information.
        """
        print('*** Trip Chain Information ***')
        print('number of EV:', len(self.ev_sample))
        print('number of trips:', len(self.ev_trip))
        print('en:', self.en)
        print('prob_en:', self.prob_en)
        print('snr:', self.snr)
        print('rate:', self.rate)
        print('rate_name:', self.rate_name)
        print('location_name:', self.location_name)

        print('*** Charging Demand Parameters ***')
        print('home_price:', self.home_price)
        print('simulated days:', self.simulated_days)
        print('L_available:', self.L_available)
        print('pub_price:', self.pub_price)
        print('zzone:', self.zzones)
        print('num_zzone:', self.num_zone)

        print('*** Charging Preference Parameters ***')
        print('charging behavior:', self.charge_behave)
        print('beta_SOC:', self.beta_SOC)
        print('beta_R:', self.beta_R)
        print('beta_delta_SOC:', self.beta_delta_SOC)
        print('beta_0:', self.beta_0)
        print('beta_SOC_0:', self.beta_SOC_0)
        print('beta_cost:', self.beta_cost)
        print('lambda:', self.lbd)

        print('*** Charging Placement Parameters ***')
        print('year:', self.year)
        print('discount_rate:', self.discount_rate)
        print('station efficiency:', self.station_efficiency)
        print('electricity cost:', self.electricity_cost)
        print('L2_buy_cost:', self.L2_buy_cost)
        print('L2_build_cost:', self.L2_build_cost)
        print('DCFC_buy_cost:', self.DCFC_buy_cost)
        print('DCFC_build_cost:', self.DCFC_build_cost)









