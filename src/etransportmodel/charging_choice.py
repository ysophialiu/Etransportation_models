import numpy as np
from etransportmodel.trip_data import TripData


class ChargingChoice():
    def __init__(self, tripData: TripData):
        """Initializes the Charging Choice module with some parameters.

        Args:
            tripData (TripData): TripData module containing trip related parameters 
        """
        self.trip = tripData


    def V_SOC(self, SOC_a: float) -> float:
        """Calculates the indirect utility for the end of trip State of Charge.

        Args:
            SOC_a (float): the value of the end of trip State of Charge, as a percent (0 to 1)

        Returns:
            float: indirect utility for the end of trip SOC
        """        
    
        if SOC_a == 1:
            V_SOC = - self.trip.beta_SOC*10
        elif SOC_a == 0:
            V_SOC =   self.trip.beta_SOC*10
        else:
            V_SOC = self.trip.beta_SOC * np.log((1-SOC_a)/((1/self.trip.beta_SOC_0-1)*SOC_a))
        
        return V_SOC


    def V_rate(self, rrate: float) -> float:
        """Calculates the utility for the charging rate of the selected charging mode

        Args:
            rrate (float): the charging rate of the selected charging mode

        Returns:
            float: indirect utility of charging
        """        
        V_rate = self.trip.beta_R * (rrate - self.trip.rate[0])
        return V_rate


    def V_d_SOC(self, SOC_b: float) -> float:
        """Calculates the utility derived from the change in the State of Charge

        Args:
            SOC_b (float): the change in State of Charge between the start and end of the trip

        Returns:
            float: marginal utility of the change in State of Charge between the start and end of the trip
        """        
        V_d_SOC = self.trip.beta_delta_SOC * (1 - (SOC_b - 1)**2)
        return V_d_SOC


    def cost_home(self, home_price: float, delta_SOC_i: float, Enn: float) -> float:
        """The cost of charging an electric vehicle at home.

        Args:
            home_price (float): price of electricity at home in $/kWh
            delta_SOC_i (float): change in percent of State of Charge over this charging session
            Enn (float): battery capacity of the electric vehicle simulated

        Returns:
            float: _description_
        """        
        cost_home = home_price * delta_SOC_i * Enn 
        return cost_home


    def V_cost(self, cost_a: float, cost_home: float) -> float:
        """Calculates the indirect utility derived from the cost of charging.

        Args:
            cost_a (float): cost of charging in the selected charging mode
            cost_home (float): cost of charging at home

        Returns:
            float: indirect utility of charging cost
        """        
        V_cost = -self.trip.beta_cost * (cost_a - cost_home) 
        return V_cost
    
    
    def charging_choice(self, SOC_l, d_time, Enn, L_available, pubprice):
        """Calculates the total indirect utility of charging using the selected charging mode

        Args:
            SOC_l (_type_): current State of Charge level
            d_time (_type_): driving time by the EV
            Enn (_type_): battery capacity of the EV
            L_available (_type_): which levels of charging are available (length 3 list) 
            pubprice (_type_):  electricity price in public in $/kWh

        Returns:
            draw: randomly drawn charging level choice based on the modelled probabilities
            p_l: probability for each charging level being chosen from the model and parameters 
        """

    
        ## indirect utility of all 4 charging mode [0,L1,L2,L3]
        
        # charge SOC of L1, L2, L3
        SOC_1 = [1-SOC_l]*3
        SOC_2 = np.array(self.trip.rate)*(d_time/Enn)
        SOC_3 = tuple(zip(SOC_1,SOC_2))
        delta_SOC = np.array([min(i) for i in SOC_3])
        price = np.array([self.trip.home_price, pubprice, pubprice])
        
        # cost of L1, L2, L3
        cost_l = np.multiply(delta_SOC*Enn,price)

        # indirect utility of all charging mode [0,L1,L2,L3]
        V = [0]*4
        V_r = [0]*3
        V_d_s = [0]*3
        V_c_home = [0]*3
        V_c = [0]*3
        for i in range(3):
            V_r[i] = self.V_rate(self.trip.rate[i])
            V_d_s[i] = self.V_d_SOC(delta_SOC[i])
            V_c_home[i] = self.cost_home(self.trip.home_price,delta_SOC[i],Enn)
            V_c[i] = self.V_cost(cost_l[i],V_c_home[i])

            V[i+1] = self.trip.beta_0 + self.V_SOC(SOC_l) + V_r[i] + V_d_s[i] + V_c[i]        
        
        e_V = np.exp([self.trip.lbd**(-1) * i for i in V]) 
        for i in range(len(L_available)):
            if L_available[i] == 0:
                e_V[i+1] = 0
            
        sum_e_V = sum(e_V)
        
        p_l = e_V/sum_e_V

        draw = np.random.choice(range(4), 1, p=p_l)
        return draw, p_l 
    