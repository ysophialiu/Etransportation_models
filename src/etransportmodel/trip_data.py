import geopandas as gpd
import pickle
import numpy as np
import torch
import pandas as pd
from scipy.stats import truncnorm

from etransportmodel.constants import DEFAULT_VALUES

"""
Description of file.
"""
class TripData():
    def __init__(
            self,
            shapefile = DEFAULT_VALUES['shapefile'], 
            tripfile = DEFAULT_VALUES['tripfile'],
            en = None,
            cn = None,
            prob_en = None,
            snr = None,
            charge_behave = DEFAULT_VALUES['charge_behave'],
            rate = None,
            rate_name = None,
            location_name = None,
            home_price = DEFAULT_VALUES['home_price'],
            D = DEFAULT_VALUES['D'],
            test_pub_price = DEFAULT_VALUES['test_pub_price'],
            L_available = None,
            pub_price = DEFAULT_VALUES['pub_price'],
            zzones = DEFAULT_VALUES['zzones'],
            year = DEFAULT_VALUES['year'], # life-cycle time
            discount_rate = DEFAULT_VALUES['discount_rate'],
            scale_to_year = DEFAULT_VALUES['scale_to_year'],
            station_efficiency = DEFAULT_VALUES['station_efficiency'],
            electricity_cost = DEFAULT_VALUES['electricity_cost'], # in $/kWh  U.S average 0.07 for industry; 0.11 for commercial
            L2_buy_cost = DEFAULT_VALUES['L2_buy_cost'], # in $
            L2_build_cost = DEFAULT_VALUES['L2_build_cost'], # in $
            DCFC_buy_cost = DEFAULT_VALUES['DCFC_buy_cost'], # in $
            DCFC_build_cost = DEFAULT_VALUES['DCFC_build_cost'], # in $
        ):

        self.initialize_trip_data(shapefile, tripfile, en, cn, prob_en, snr, charge_behave)
        self.initialize_charging_choice(rate, rate_name, location_name, home_price, D, test_pub_price, L_available)
        self.intiialize_charging_demand(pub_price, zzones)
        self.initialize_charging_placement(year, discount_rate, scale_to_year, station_efficiency, electricity_cost, 
                                           L2_buy_cost, L2_build_cost, DCFC_buy_cost, DCFC_build_cost)


    """
    Documentation.
    """
    def initialize_trip_data(self, shapefile, tripfile, en, cn, prob_en, snr, charge_behave):
        self.shapefile = gpd.read_file(shapefile)
        self.ev_trip = pickle.load(open(tripfile, "rb")) 
        self.ev_sample = self.ev_trip["EV_list"].unique() # EV ID list
        self.ev_n_trips = self.ev_trip['EV_list'].value_counts() # EV ID & number of trips, return a pandas series: EV_N_trips[EV.N] = it's number of trips

        ###__________define EV battery capacity & energy rate by distribution
        # define possible energy of EV kWh
        # Reference: Tesla (60-100kWh); Nissan Leaf (40-60kWh)
        if en is None:
            en = DEFAULT_VALUES['en']
        self.en = en

        # define energy consumption rate: kWh/mile 
        #Reference: Tesla (0.33-0.38kwh/mi); Audi (0.43kWh/mi)
        if cn is None:
            cn = DEFAULT_VALUES['cn']
        self.cn = cn

        # define probability distriburion of each En value
        if prob_en is None:
            prob_en = DEFAULT_VALUES['prob_en']
        self.prob_en = prob_en

        if snr is None:
            snr = DEFAULT_VALUES['snr']
        self.snr = snr # list used in the dictionary

        self.charge_behave = charge_behave

        # list of En and Cn
        self.Scenario = np.random.choice(self.snr, len(self.ev_sample), p=self.prob_en)
        # create En Cn for the EV list
        self.en_v = [self.en.get(n) for n in self.Scenario]
        self.cn_v = [self.cn.get(n) for n in self.Scenario]
        self.en_v = dict(zip(self.ev_sample, self.en_v))
        self.cn_v = dict(zip(self.ev_sample, self.cn_v))

        self.mean, self.sd, self.low, self.upp = self.Init_SOC(self.charge_behave)
        self.SOC_int = self.get_truncated_normal(self.mean, self.sd, self.low, self.upp).rvs(size=len(self.ev_sample))


    """
    Documentation.
    """
    def initialize_charging_choice(self, rate, rate_name, location_name, home_price, D, test_pub_price, L_available):
        """
        define charging rate kw for level 1,2,3
        Reference: SAE J 1772 charging specification
        level 1: 120 volt (V) AC charge. 1.4-1.9kW
        level 2: 240 volt (V) AC charge. 3.7-6.6-19.2kW
        level 3: 480 volt (V) DC charge. 24-36-90-240 kW 
        """
        if rate is None:
            rate = np.array([3.6,6.2,150]) # kW
        self.rate = rate

        if rate_name is None:
            rate_name = ['h2','l2','l3']
        self.rate_name = rate_name

        if location_name is None:
            location_name = ['home','work','public']
        self.location_name = DEFAULT_VALUES['location_name']

        """ 
        define price $/kwh for level 1,2,3
        NREL: 0.11$/kWh  https://afdc.energy.gov/fuels/electricity_charging_home.html
        """
        self.home_price = home_price

        self.D = D  # simulate D days, D>1
        self.test_pub_price = test_pub_price

        if L_available is None:
            L_available = DEFAULT_VALUES['L_available']
        self.L_available = L_available

        self.charging_behavior_parameter(self.charge_behave)
    
    
    """
    Documentation.
    """
    def intiialize_charging_demand(self, pub_price, zzones):
        self.pub_price = pub_price
        self.zzones = zzones
        self.num_zone = len(self.shapefile[self.zzones].unique())

        zone_array = np.sort(self.shapefile[self.zzones].unique()).tolist()
        self.shapefile['new_zone_name'] = [zone_array.index(x)+1 for x in self.shapefile[self.zzones]]

        join_zones = pd.DataFrame(self.shapefile[['OBJECTID','new_zone_name']]).set_index('OBJECTID')
        self.ev_trip = self.ev_trip.join(join_zones, on="d_taz")
        self.ev_trip['d_taz']=self.ev_trip['new_zone_name']
        self.ev_trip = self.ev_trip.drop(columns =['new_zone_name'])


    """
    Documentation.
    """
    def initialize_charging_placement(self, year, discount_rate, scale_to_year, station_efficiency, electricity_cost, 
                                      L2_buy_cost, L2_build_cost, DCFC_buy_cost, DCFC_build_cost):
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


    """
    Documentation.
    """
    def Init_SOC(self, cases):
        # mean, sd, low, upp
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


    """
    Documentation.
    """
    def get_truncated_normal(self, mean,sd, low, upp): #(mean, sd, low, upp):
        #mean = low + (upp-low)/2
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    

    """
    Documentation.
    """
    def charging_behavior_parameter(self, cases: str) -> None:
        # beta_SOC,beta_R,beta_delta_SOC,beta_0,beta_cost,beta_SOC_0,lbd
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


   # Find cleaner way to print parameters
    def summary(self):
        print('* charging choice parameters:')
        print('number of EV',len(self.ev_sample))
        print('number of trips',len(self.ev_trip))
        print('en', self.en)
        print('prob_en', self.prob_en)
        print('snr', self.snr)
        print('charge_behave', self.charge_behave)
        print('rate', self.rate)
        print('rate_name', self.rate_name)
        print('location_name', self.location_name)
        print('home_price', self.home_price)
        print('D', self.D)
        print('test_pub_price', self.test_pub_price)
        print('L_available', self.L_available)
        print('beta_SOC:', self.beta_SOC)
        print('beta_R:', self.beta_R)
        print('beta_delta_SOC:', self.beta_delta_SOC)
        print('beta_0:', self.beta_0)
        print('beta_SOC_0:', self.beta_SOC_0)
        print('beta_cost:', self.beta_cost)
        print('lambda:', self.lbd)
        print('pub_price', self.pub_price)
        print('zzone', self.zzone)
        print('num_zzone', self.num_zone)




