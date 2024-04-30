import pandas as pd
import torch

from etransportmodel.trip_data import TripData
from etransportmodel.charging_demand import ChargingDemand


class ChargingPlacement(ChargingDemand):
    def __init__(self, tripData: TripData):
        """Initializes the Charging Placement module with some parameters.

        Args:
            tripData (TripData): TripData module containing trip related parameters 
        """      
        self.trip = tripData


    def daily_revenue(self, ciz: list) -> float:
        """Daily average public charging revenue: compute total revenue of the last effective simulation day

        Args:
            ciz (list): list of work and public chargers available

        Returns:
            float: total revenue
        """        
        ress = ChargingDemand.sim_demand_faster(self, ciz) 
        E_taz = pd.DataFrame.from_dict({(i,j,k): ress[1][i][j][k]
                            for i in ress[1].keys() 
                            for j in ress[1][i].keys()
                            for k in ress[1][i][j].keys()},
                        orient='index')
        
        E_taz.reset_index(inplace=True)  
        E_taz = E_taz.rename(columns={"level_0": "day", "level_1": "TAZ", "level_2": "charger type"})
        
        E_tazD = E_taz[E_taz['day'] == self.trip.simulated_days-1] # the day^th energy (day is numerated as:0,1,...,D-1)
    
        E_work = sum(E_tazD['work'])
        E_public = sum(E_tazD['public'])
        E_non_home = E_work + E_public
        
        revenue = E_non_home* self.trip.pub_price 
        
        return revenue 

    
    def daily_revenue_BOTorch(self, inds: list) -> torch.Tensor:
        """Daily average public charging revenue: compute total revenue of the last effective simulation day

        Args:
            inds (list): list of ciz

        Returns:
            torch.Tensor: tensor list of revenues
        """        
        
        rst = [] # store result
        for x in inds:
            
            ress =  ChargingDemand.sim_demand_faster(self, x) #run simulation
        
            E_taz = pd.DataFrame.from_dict({(i,j,k): ress[1][i][j][k]
                                for i in ress[1].keys() 
                                for j in ress[1][i].keys()
                                for k in ress[1][i][j].keys()},
                            orient='index')

            E_taz.reset_index(inplace=True)  
            E_taz = E_taz.rename(columns={"level_0": "day", "level_1": "TAZ", "level_2": "charger type"})

            E_tazD = E_taz[E_taz['day'] == self.trip.simulated_days-1] # the day^th energy (day is numerated as:0,1,...,D-1)

            E_work = sum(E_tazD['work'])
            E_public = sum(E_tazD['public'])
            E_non_home = E_work + E_public

            revenue = E_non_home* self.trip.pub_price #all_inputes[0]
            
            rst.append(revenue)
            
        return torch.tensor(rst)


    def daily_demand_BOTorch(self, inds: list) -> torch.Tensor:
        """Daily average public charging demand: compute total demand of the last effective simulation day

        Args:
            inds (list): list of ciz

        Returns:
            torch.Tensor: tensor list of demands
        """        
        
        rst = [] # store result
        for x in inds:
            
            ress =  ChargingDemand.sim_demand_faster(self, x) #run simulation
        
            E_taz = pd.DataFrame.from_dict({(i,j,k): ress[1][i][j][k]
                                for i in ress[1].keys() 
                                for j in ress[1][i].keys()
                                for k in ress[1][i][j].keys()},
                            orient='index')

            E_taz.reset_index(inplace=True)  
            E_taz = E_taz.rename(columns={"level_0": "day", "level_1": "TAZ", "level_2": "charger type"})

            E_tazD = E_taz[E_taz['day'] == self.trip.simulated_days-1] # the day^th energy (day is numerated as:0,1,...,D-1)

            E_work = sum(E_tazD['work'])
            E_public = sum(E_tazD['public'])
            E_non_home = E_work + E_public
            
            rst.append(E_non_home)
            
        return torch.tensor(rst)


    def energy_sum(self, ciz: list, p_price: float, day: int) -> tuple:
        """Compute total energy used in public.

        Args:
            ciz (list): list of work and public chargers available
            p_price (float): public price of electricity in $ kwH/h
            day (int): simulation day

        Returns:
            tuple: _description_
        """          
        ress = ChargingDemand.sim_demand(self, ciz, p_price)
        
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


    def Optimization_function(self, inds: list) -> float:
        """Calculation Net Present Value (NPV)

        Args:
            inds (list): list of ciz

        Returns:
            float: expected net present value
        """
        
        inds = torch.clamp(inds, min=0, max=None)
        
        average_daily_demand = self.daily_demand_BOTorch(inds)

        year_revenue = self.trip.scale_to_year * self.trip.pub_price * average_daily_demand 

        year_electricity_cost = self.trip.scale_to_year * self.trip.electricity_cost * average_daily_demand / self.trip.station_efficiency
   
        initial_invest_cost = sum((self.trip.invest_cost*inds).t()).t()

        year_m_o_cost = 0.1 * initial_invest_cost
        
        year_benefit = year_revenue - year_electricity_cost - year_m_o_cost

        NPV = self.trip.present_worth_factor * year_benefit - initial_invest_cost
        return NPV
    