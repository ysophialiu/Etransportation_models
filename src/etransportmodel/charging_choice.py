import numpy as np
from etransportmodel.trip_data import TripData

"""
Description of file.
"""
class ChargingChoice():
    """Initializes the Charging Choice module with some parameters

    Keyword arguments:
    tripData -- TripData module containing trip related parameters 
    """
    def __init__(self, tripData: TripData):
        self.trip = tripData

    """Calculates the indirect utility for the end of trip State of Charge
    
    Keyword arguments:
    SOC_a -- the value of the end of trip State of Charge, as a percent (0 to 1)
    """
    def charging_choice_parameters(self):
        print('* charging choice parameters:')
        print('beta_SOC:', self.trip.beta_SOC)
        print('beta_R:', self.trip.beta_R)
        print('beta_delta_SOC:', self.trip.beta_delta_SOC)
        print('beta_0:', self.trip.beta_0)
        print('beta_SOC_0:', self.trip.beta_SOC_0)
        print('beta_cost:', self.trip.beta_cost)
        print('lambda:', self.trip.lbd)
        print('* charging choice plot parameters')
        print('not-home charging price ($):',self.trip.test_pub_price)
        print('available charging type (home, non-home L2, non_home DCFC):', self.trip.L_available)


    """
    Documentation.
    """
    def V_SOC(self, SOC_a):
        if SOC_a == 1:
            V_SOC = - self.trip.beta_SOC*10 #replace np.inf because of calculation issue
        elif SOC_a == 0:
            V_SOC =   self.trip.beta_SOC*10 #np.inf
        else:
            V_SOC = self.trip.beta_SOC * np.log((1-SOC_a)/((1/self.trip.beta_SOC_0-1)*SOC_a))
        
        return V_SOC


    """Calculates the utility for the charging rate of the selected charging mode

    Keyword arguments:
    rrate -- the charging rate of the selected charging mode
    """
    def V_rate(self, rrate):
        V_rate = self.trip.beta_R * (rrate-self.trip.rate[0])
        return V_rate


    """Calculates the utility derived from the change in the State of Charge
    
    Keyword arguments:
    SOC_b -- the change in State of Charge between the start and end of the trip
    """
    def V_d_SOC(self, SOC_b):
        V_d_SOC = self.trip.beta_delta_SOC * (1 - (SOC_b - 1)**2)
        return V_d_SOC


    """
    The cost of charging an electric vehicle at home

    Keyword arguments:
    home_price: price of electricity at home in $/kWh
    delta_SOC_i: change in percent of State of Charge over this charging session 
    Enn: battery capacity of the electric vehicle simulated
    """
    def cost_home(self, home_price,delta_SOC_i,Enn):
        cost_home = home_price*delta_SOC_i*Enn 
        return cost_home


    """
    Calculates the indirect utility derived from the cost of charging 

    Keyword arguments:
    cost_a: cost of charging in the selected charging mode
    cost_home: cost of charging at home
    """
    def V_cost(self, cost_a,cost_home):
        V_cost = -self.trip.beta_cost * (cost_a - cost_home) 
        return V_cost
    
    
    """
    Calculates the total indirect utility of charging using the selected charging mode

    Keyword arguments:
    SOC_l: current State of Charge level
    d_time: driving time by the EV
    Enn: battery capacity of the EV
    L_available: which levels of charging are available (length 3 list) 
    pubprice: electricity price in public in $/kWh

    Returns: 
    draw: randomly drawn charging level choice based on the modelled probabilities
    p_l: probability for each charging level being chosen from the model and parameters 
    """
    def charging_choice(self, SOC_l, d_time, Enn, L_available, pubprice):
    
        ## indirect utility of all 4 charging mode [0,L1,L2,L3]
        
        # charge SOC of L1, L2, L3
        SOC_1 = [1-SOC_l]*3
        SOC_2 = self.trip.rate*(d_time/Enn)
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
        
        #print('V_c:',V_c)
        #print('V_d_s:',V_d_s)    
        #print('V_r:',V_r)
        #print('V:',V)   
        
        # e^V
        e_V = np.exp([self.trip.lbd**(-1) * i for i in V]) 
        for i in range(len(L_available)):
            if L_available[i] == 0:
                e_V[i+1] = 0
            
        #print('e_V',e_V)
        sum_e_V = sum(e_V)
        
        p_l = e_V/sum_e_V
        #print('p',p_l)
        #print('probability per L',p_l)
        draw = np.random.choice(range(4), 1, p=p_l)
        return draw, p_l #, V, SOC_2,  V_d_s, V_c, delta_SOC
    