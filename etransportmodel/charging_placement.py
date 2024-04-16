import pandas as pd
import torch

from etransportmodel.trip_data import TripData
from etransportmodel.charging_demand import ChargingDemand

"""
Description of file.
"""
class ChargingPlacement(ChargingDemand):
    def __init__(self, tripData: TripData):
        self.trip = tripData

    """
    Documentation.
    """
    def daily_revenue(self, ciz):#(ciz,p_price): # run_sim_demand is the return result of the previous function; day is the nth day 
        
        ress = ChargingDemand.sim_demand_faster(ciz) #sim_demand(ciz)#run_sim_demand # run_sim_demand = sim_demand(ciz)
        
        E_taz = pd.DataFrame.from_dict({(i,j,k): ress[1][i][j][k]
                            for i in ress[1].keys() 
                            for j in ress[1][i].keys()
                            for k in ress[1][i][j].keys()},
                        orient='index')
        
        E_taz.reset_index(inplace=True)  
        E_taz = E_taz.rename(columns={"level_0": "day", "level_1": "TAZ", "level_2": "charger type"})
        
        E_tazD = E_taz[E_taz['day'] == self.trip.D-1] # the day^th energy (day is numerated as:0,1,...,D-1)
    
        #E_home = sum(E_tazD['home'])
        E_work = sum(E_tazD['work'])
        E_public = sum(E_tazD['public'])
        E_non_home = E_work + E_public
        #E_all = E_home + E_work + E_public
        
        revenue = E_non_home* self.trip.pub_price #all_inputes[0]
        
        return revenue # -revenue  if use skopt

    
    """
    Documentation.
    # daily average public charging revenue: compute total revenue of the last effective simulation day
    """
    def daily_revenue_BOTorch(self, inds):#inds is a list of [ciz]
        
        rst = [] # store result
        for x in inds:
            
            ress =  ChargingDemand.sim_demand_faster(x) #run simulation
        
            E_taz = pd.DataFrame.from_dict({(i,j,k): ress[1][i][j][k]
                                for i in ress[1].keys() 
                                for j in ress[1][i].keys()
                                for k in ress[1][i][j].keys()},
                            orient='index')

            E_taz.reset_index(inplace=True)  
            E_taz = E_taz.rename(columns={"level_0": "day", "level_1": "TAZ", "level_2": "charger type"})

            E_tazD = E_taz[E_taz['day'] == self.trip.D-1] # the day^th energy (day is numerated as:0,1,...,D-1)

            E_work = sum(E_tazD['work'])
            E_public = sum(E_tazD['public'])
            E_non_home = E_work + E_public

            revenue = E_non_home* self.trip.pub_price #all_inputes[0]
            
            rst.append(revenue)
            
        return torch.tensor(rst)


    """
    Documentation.
    daily average public charging revenue: compute total revenue of the last effective simulation day
    """
    def daily_demand_BOTorch(self, inds):#inds is a list of [ciz]
        
        rst = [] # store result
        for x in inds:
            
            ress =  ChargingDemand.sim_demand_faster(x) #run simulation
        
            E_taz = pd.DataFrame.from_dict({(i,j,k): ress[1][i][j][k]
                                for i in ress[1].keys() 
                                for j in ress[1][i].keys()
                                for k in ress[1][i][j].keys()},
                            orient='index')

            E_taz.reset_index(inplace=True)  
            E_taz = E_taz.rename(columns={"level_0": "day", "level_1": "TAZ", "level_2": "charger type"})

            E_tazD = E_taz[E_taz['day'] == self.trip.D-1] # the day^th energy (day is numerated as:0,1,...,D-1)

            E_work = sum(E_tazD['work'])
            E_public = sum(E_tazD['public'])
            E_non_home = E_work + E_public

            #demand = E_non_home* pub_price #all_inputes[0]
            
            rst.append(E_non_home)
            
        return torch.tensor(rst)


    """
    Documentation.
    compute total energy
    """
    def energy_sum(self, ciz,p_price,day): # run_sim_demand is the return result of the previous function; day is the nth day 
        
        ress = ChargingDemand.sim_demand(ciz,p_price)#run_sim_demand # run_sim_demand = sim_demand(ciz)
        
        E_taz = pd.DataFrame.from_dict({(i,j,k): ress[1][i][j][k]
                            for i in ress[1].keys() 
                            for j in ress[1][i].keys()
                            for k in ress[1][i][j].keys()},
                        orient='index')
        
        E_taz.reset_index(inplace=True)  
        E_taz = E_taz.rename(columns={"level_0": "day", "level_1": "TAZ", "level_2": "charger type"})
        
        E_tazD = E_taz[E_taz['day'] == day] # the day^th energy
    
        E_home = sum(E_tazD['home'])
        E_work = sum(E_tazD['work'])
        E_public = sum(E_tazD['public'])
        E_non_home = E_work + E_home
        E_all = E_home + E_work + E_public
        
        # non home percent
        pcent = E_non_home/E_all


        return E_non_home, E_all, pcent


    """
    Documentation.
    Declare NPV.
    """
    def Optimization_function(self, inds):
        
        inds = torch.clamp(inds, min=0, max=None) # clamp negative values to 0
        
        average_daily_demand = self.daily_demand_BOTorch(inds)
        #print('average_daily_demand',average_daily_demand)
        
        year_revenue = self.trip.scale_to_year * self.trip.pub_price * average_daily_demand 
        #print('year_revenue',year_revenue)
        
        year_electricity_cost = self.trip.scale_to_year * self.trip.electricity_cost * average_daily_demand / self.trip.station_efficiency
        #print('year_electricity_cost',year_electricity_cost)    
        #print(inds)    
        initial_invest_cost = sum((self.trip.invest_cost*inds).t()).t()
        #print('initial_invest_cost',initial_invest_cost)
        
        year_m_o_cost = 0.1 * initial_invest_cost
        #print('year_m_o_cost',year_m_o_cost)
        
        year_benefit = year_revenue - year_electricity_cost - year_m_o_cost
        #print('year_benefit',year_benefit)
        
        NPV = self.trip.present_worth_factor * year_benefit - initial_invest_cost
        return NPV
    