import warnings
import torch
import numpy as np
 
from botorch import fit_gpytorch_model
from botorch.models import FixedNoiseGP
from botorch.acquisition.analytic import ExpectedImprovement,UpperConfidenceBound,ProbabilityOfImprovement, PosteriorMean
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.exceptions import BadInitialCandidatesWarning

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from etransportmodel.charging_placement import ChargingPlacement
from etransportmodel.trip_data import TripData


class OptimizationSolvers(ChargingPlacement):
    def __init__(self, tripData: TripData, low_dim=5, x_l_bound=-1, x_u_bound=1, y_variance=0.1, max_station_capacity=5):
        """Initializes the Optimization Solvers module with some parameters.

        Args:
            tripData (TripData): TripData module containing trip related parameters 
            low_dim (int, optional): Dimensionality of the low-dimensional search space in REMBO. Defaults to 5.
            x_l_bound (int, optional): lower bound of embedding search space. Defaults to -1.
            x_u_bound (int, optional): upper bound of embedding search space. Defaults to 1.
            y_variance (float, optional): Variance of the observation noise model used in Gaussian processes. Defaults to 0.1.
            max_station_capacity (int, optional): Maximum capacity of EV charging stations that can be modeled. Defaults to 5.
        """
        self.trip = tripData
        self.low_dim = low_dim
        self.dim = 2 * self.trip.num_zone
        self.initialize_optimization_solvers(x_l_bound, x_u_bound, y_variance, max_station_capacity)


    def initialize_optimization_solvers(self, x_l_bound: int, x_u_bound: int, y_variance: float, max_station_capacity: int):
        """Initialize solver model information.

        Args:
            x_l_bound (int): lower bound of embedding search space
            x_u_bound (int): upper bound of embedding search space
            y_variance (float): y variance
            max_station_capacity (int): maximum number of stations allowed at each location
        """        
        max_EV_per_zone = self.create_ev_zones(max_station_capacity)
        self.x_real_u_bound = np.append(list(max_EV_per_zone.values()),list(max_EV_per_zone.values()))
        self.x_real_u_bound = torch.tensor(self.x_real_u_bound,dtype = self.trip.ddtype)
        self.x_real_l_bound = torch.tensor([0]*self.dim,dtype = self.trip.ddtype)
        self.x_real_bound = torch.tensor([self.x_real_l_bound.tolist(),self.x_real_u_bound],dtype = self.trip.ddtype)

    
        self.x_l_bound = x_l_bound
        self.x_u_bound = x_u_bound

        self.y_l_bound = -np.sqrt(self.low_dim)
        self.y_u_bound = np.sqrt(self.low_dim)

        self.y_variance = y_variance

        self.x_scale_bound = (torch.tensor([[x_l_bound],[x_u_bound]])).repeat(1,self.dim).to(self.trip.ddtype)
        self.y_bound = (torch.tensor([[self.y_l_bound],[self.y_u_bound]])).repeat(1,self.low_dim).to(self.trip.ddtype) 


    def create_ev_zones(self, max_station_capacity: int) -> dict:
        """Creates the mapping for the maximum number of stations allowed at each location.
        
        Args:
            max_station_capacity (int): maximum number of stations allowed

        Returns:
            dict: max number of stations allowed to be at each location
        """
        N_map = {}

        for i in self.trip.shapefile['new_zone_name']:
            N_map[i] = {}
            for t in np.arange(0,48.1,0.5):
                N_map[i][round(t, 2)] = 0

        for _, row in self.trip.ev_trip.iterrows():
            d_purpose = row['d_purpose']
            if d_purpose != 'Home':
                tazz = row['d_taz']
                a_time = row['end_period']
                b_time = row['end_period'] + row['dwell_time']*2
                
                for t in np.arange(1,48.1,0.5):
                    t = round(t, 2)
                    if t == a_time:
                        N_map[tazz][t] = N_map[tazz][t]+1

                for t in np.arange(1,48.1,0.5):
                    t = round(t, 2)
                    if t >= a_time and t <= b_time:
                        N_map[tazz][t] = N_map[tazz][t]+1

        max_EV_per_zone= {}
        for i in self.trip.shapefile['new_zone_name']:
            max_EV_per_zone[i] =  round( 0.1 * max(N_map[i].values())) 
            if max_EV_per_zone[i] == 0:
                max_EV_per_zone[i] = max_EV_per_zone[i] + 0.1
            if max_EV_per_zone[i] >= max_station_capacity:
                max_EV_per_zone[i] = max_station_capacity
        
        return max_EV_per_zone


    
    ### BAYESIAN OPTIMIZATION ###
    
    def gen_high_dimension_variable(self, nn: int) -> torch.Tensor:
        """Generate y, with 0.

        Args:
            nn (int): number of random y, nn <= 10
        """        
        rand_yy = []
        rand_yy.append([0] * self.dim)
        for _ in range(nn - 1):
            test_y = np.random.uniform(self.x_real_l_bound.tolist(), self.x_real_u_bound.tolist()).astype(int)
            rand_yy.append(test_y)
        return torch.FloatTensor(rand_yy).to(self.trip.ddtype)
    

    def generate_initial_data_BO(self, n: int) -> tuple: 
        """Generate initial data for Bayesian Optimization.

        Args:
            n (int): number of initial value to generate
        """        
        train_x = self.gen_high_dimension_variable(n)
        exact_obj = ChargingPlacement.Optimization_function(self, train_x).unsqueeze(-1).to(self.trip.ddtype)
        best_observation_value = exact_obj.max().item()
        best_observation_x = train_x[exact_obj.argmax().item()]

        return train_x, exact_obj, best_observation_value, best_observation_x


    def get_next_points_BO(self, init_x , init_y, best_init_y, bounds_init_x, BATCH_SIZE) -> torch.Tensor:
        """Get new data point for Bayesian Optimization

        Args:
            init_x (torch.Tensor): Current input data points.
            init_y (torch.Tensor): Outputs corresponding to init_x.
            best_init_y (float): The best observed output value.
            bounds_init_x (torch.Tensor): Bounds for the inputs.
            BATCH_SIZE (int): Number of new candidate points to generate.
        """
        BO_bound = (torch.tensor([[0],[1]])).repeat(1,self.dim).to(self.trip.ddtype)
        
        norm_init_x = normalize(init_x, bounds=bounds_init_x)
        
        mean_init_y = init_y.mean().item()
        std_init_y = init_y.std().item()
        
        norm_init_y = (init_y-mean_init_y)/std_init_y 
        norm_init_Y_var = torch.full_like(norm_init_y, self.y_variance)
        norm_best_init_y = norm_init_y.max().item()
        
        single_model = FixedNoiseGP(norm_init_x,norm_init_y,norm_init_Y_var)
    
        mml = ExactMarginalLogLikelihood(single_model.likelihood,single_model) 
        fit_gpytorch_model(mml)

        EI = ExpectedImprovement(model = single_model, best_f = norm_best_init_y)

        norm_candidates, _ = optimize_acqf(
                        acq_function = EI,
                        bounds = BO_bound,
                        q = BATCH_SIZE,
                        num_restarts = 50,
                        raw_samples = 1000)
            
        candidates = unnormalize(norm_candidates, bounds=bounds_init_x)

        return candidates.int().to(self.trip.ddtype)

    
    def BO_run(self, N_initial, BATCH_SIZE, N_BATCH, N_TRIALS) -> tuple:
        """Bayesian Optimization Run

        Args:
            N_initial (int): Number of initial data points.
            BATCH_SIZE (int): Number of candidates in each batch.
            N_BATCH (int): Number of batches to run.
            N_TRIALS (int): Number of trials to perform.
        """
        
        warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        best_observed_all = []

        best_observed_all_x = []

        for trial in range(1, N_TRIALS + 1):


            print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
            best_observed = []
            best_observed_x = []

            init_x, init_y, best_init_y, best_init_x = self.generate_initial_data_BO(N_initial)

            best_observed.append(best_init_y)
            best_observed_x.append(best_init_x)

            # run N_BATCH rounds of BayesOpt after the initial random batch
            for iteration in range(1, N_BATCH + 1):    

                new_candidates = self.get_next_points_BO(init_x, init_y, best_init_y, self.x_real_bound, BATCH_SIZE)

                new_results = ChargingPlacement.Optimization_function(self, new_candidates).unsqueeze(-1)

                init_x = torch.cat([init_x,new_candidates])
                init_y = torch.cat([init_y,new_results])

                best_init_y = init_y.max().item()
                best_init_x = init_x[init_y.argmax().item()]

                best_observed.append(best_init_y)
                best_observed_x.append(best_init_x)         

            best_observed_all.append(best_observed)
            best_observed_all_x.append(best_observed_x)

        return best_observed_all, best_observed_all_x, init_x, init_y



    def BO_add_iter(self, N_BATCH_, BATCH_SIZE_, x_real_bound_, RUN_RESULT) -> tuple:
        """Run more iterations of Bayesian Optimization after intial batch run.

        Args:
            N_BATCH_ (int): Number of additional batches to run.
            BATCH_SIZE_ (int): Number of candidates in each additional batch.
            x_real_bound_ (torch.Tensor): Boundaries for the input space.
            RUN_RESULT (tuple): Contains the results of a previous Bayesian Optimization run.
        """
        best_observed_all_, best_observed_all_x_, init_x_, init_y_ = RUN_RESULT
        best_init_y_ = init_y_.max().item()
        
        for _ in range(1, N_BATCH_ + 1):    

            new_candidates = self.get_next_points_BO(init_x_, init_y_, best_init_y_, x_real_bound_, BATCH_SIZE_)
            new_results = ChargingPlacement.daily_revenue_BOTorch(self, new_candidates).unsqueeze(-1)

            init_x_ = torch.cat([init_x_,new_candidates])
            init_y_ = torch.cat([init_y_,new_results])

            best_init_y = init_y_.max().item()
            best_init_x = init_x_[init_y_.argmax().item()]

            best_observed_all_[0].append(best_init_y)
            best_observed_all_x_[0].append(best_init_x)
                
        return best_observed_all_,best_observed_all_x_, init_x_, init_y_


    def gen_projection_rembo(self, d: int, D: int) -> torch.Tensor:
        """Generate the projection matrix A as a (d x D) tensor.

        Args:
            d (int): low dimension
            D (int): high dimension
        """        
        AA = torch.randn(D,d, dtype=self.trip.ddtype)
        return AA


    def gen_low_dimension_variable(self, nn: int) -> torch.Tensor:
        """Generate y in low dimension, with 0.

        Args:
            nn (int): number of random y, nn <= 10
        """        
        rand_yy = []
        rand_yy.append([0] * self.low_dim)
        for i in range (nn-1):
            test_y = np.random.uniform(self.y_l_bound, self.y_u_bound,self.low_dim)
            rand_yy.append(test_y)
        return torch.FloatTensor(rand_yy).to(self.trip.ddtype)
    

    def low_to_high_dimension(self, AA: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        """Convert low dimension y to high dimension x

        Args:
            AA (torch.Tensor): random embedding matrix, low_dim * dim
            yy (torch.Tensor): the low dimension variable, in low_dim
        """
        scale_xx = torch.t(torch.matmul(AA,torch.t(yy)))
        scale_xx = torch.clamp(scale_xx, min=self.x_l_bound, max=self.x_u_bound)
        scale_xx_ = (scale_xx-self.x_l_bound)/(self.x_u_bound-self.x_l_bound)
        real_xx = scale_xx_*(self.x_real_u_bound - self.x_real_l_bound) + self.x_real_l_bound #scale_xx_*x_real_u_bound
        return real_xx.int().to(self.trip.ddtype)


    def generate_initial_data_REMBO(self, AA: torch.Tensor, n :int) -> tuple:
        """Generate initial data points for REMBO run

        Args:
            AA (torch.Tensor): random embedding matrix, low_dim * dim
            n (int): number of initial value we want to generate
        """
        # n is number of initial value want to generate
        gen_low = self.gen_low_dimension_variable(n)
        train_x = self.low_to_high_dimension(AA,gen_low)
        exact_obj = ChargingPlacement.Optimization_function(self, train_x).unsqueeze(-1).to(self.trip.ddtype)
        best_observation_value = exact_obj.max().item()
        best_observation_low = gen_low[exact_obj.argmax().item()]

        return gen_low,exact_obj, best_observation_value, best_observation_low #train_x.float()


    def get_next_points_REMBO(self, init_x, init_y, bounds_init_x, BATCH_SIZE):
        """Generates the next batch of points for REMBO optimization by projecting low-dimensional suggestions to high dimensions.

        Args:
            init_x (torch.Tensor): The current set of low-dimensional input data points.
            init_y (torch.Tensor): Outputs associated with init_x.
            bounds_init_x (torch.Tensor): Bounds for low-dimensional input space.
            BATCH_SIZE (int): Number of new candidates to generate.
        """   
        BO_bound = (torch.tensor([[0],[1]])).repeat(1,self.low_dim).to(self.trip.ddtype)
        
        norm_init_x = normalize(init_x, bounds=bounds_init_x) # normalize x into [0,1]
        
        mean_init_y = init_y.mean().item() # mean of y
        std_init_y = init_y.std().item()  # std of y
        
        norm_init_y = (init_y-mean_init_y)/std_init_y  # standardize y    
        norm_init_Y_var = torch.full_like(norm_init_y, self.y_variance) # y with noise    
        norm_best_init_y = norm_init_y.max().item() # best stardized y
        #print(norm_init_x,norm_init_y,norm_init_Y_var)
        
        single_model = FixedNoiseGP(norm_init_x,norm_init_y,norm_init_Y_var) # define GP: single task homoskedastic exact GP
    
        mml = ExactMarginalLogLikelihood(single_model.likelihood,single_model) # define likelihood to fit GP:The exact marginal log likelihood (MLL) for an exact Gaussian process with a Gaussian likelihood.

        fit_gpytorch_model(mml) # Fit hyperparameters of a GPyTorch model, L-BFGS-B via scipy.optimize.minimize().
        #fit_gpytorch_model(mml, optimizer=fit_gpytorch_torch) # Fit hyperparameters of a GPyTorch model, line search

        EI = ExpectedImprovement(model = single_model,best_f = norm_best_init_y) # define acquisition function by GP

        #EI = UpperConfidenceBound(model = single_model) # define acquisition function by GP


        #print('1')
        norm_candidates, _ = optimize_acqf(
                        acq_function = EI,
                        bounds = BO_bound,
                        q = BATCH_SIZE,
                        num_restarts = 50,
                        raw_samples = 1000)
            
        candidates = unnormalize(norm_candidates, bounds=bounds_init_x)

        return candidates.to(self.trip.ddtype)#, norm_candidates   # round to the lowest closet integer; change type back to 


   # define run rembo
    def REMBO_run(self, N_initial, BATCH_SIZE, N_BATCH, N_TRIALS):
        """Executes multiple rounds of REMBO to optimize the placement of EV charging stations over several trials.

        Args:
            N_initial (int): Number of initial data points.
            BATCH_SIZE (int): Number of new candidate points per batch.
            N_BATCH (int): Number of optimization batches per trial.
            N_TRIALS (int): Number of independent trials to run.
        """
        warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        best_observed_all = []

        best_observed_all_x = []

        A_all = []

        # average over multiple trials
        for trial in range(1, N_TRIALS + 1):

            A = self.gen_projection_rembo(self.low_dim,self.dim)

            A_all.append(A)


            print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
            best_observed = []
            best_observed_x = []


            # generate initial training data and initialize model
            init_x, init_y, best_init_y, best_init_x = self.generate_initial_data_REMBO(A,N_initial)
            #best_observed_high = low_to_high_dimension(A,best_init_x)

            best_observed.append(best_init_y)
            best_observed_x.append(best_init_x)

            # run N_BATCH rounds of BayesOpt after the initial random batch
            for _ in range(1, N_BATCH + 1):    

                new_candidates = self.get_next_points_REMBO(init_x, init_y, self.y_bound, BATCH_SIZE)

                new_high = self.low_to_high_dimension(A,new_candidates)

                new_results = ChargingPlacement.Optimization_function(self, new_high).unsqueeze(-1)

                init_x = torch.cat([init_x,new_candidates])
                init_y = torch.cat([init_y,new_results])

                best_init_y = init_y.max().item()
                best_init_x = init_x[init_y.argmax().item()]

                #print(new_results)

                best_observed.append(best_init_y)
                best_observed_x.append(best_init_x)              

            best_observed_all.append(best_observed)
            best_observed_all_x.append(best_observed_x)

        return best_observed_all,best_observed_all_x, init_x, init_y, A_all


    # one step of iteration (GP and acquisition), and find the next point
    def get_next_points_REMBO_ac(self, init_x, init_y, bounds_init_x, BATCH_SIZE,ac_function):
        """Selects the next points for evaluation in REMBO using different acquisition functions based on user selection.

        Args:
            init_x (torch.Tensor): The current set of low-dimensional input data points.
            init_y (torch.Tensor): Outputs associated with init_x.
            bounds_init_x (torch.Tensor): Bounds for low-dimensional input space.
            BATCH_SIZE (int): Number of new candidate points to generate.
            ac_function (str): Type of acquisition function to use ('EI', 'PI', 'UCB', 'PM').
        """

        BO_bound = (torch.tensor([[0],[1]])).repeat(1,self.low_dim).to(self.trip.ddtype)
        
        norm_init_x = normalize(init_x, bounds=bounds_init_x) # normalize x into [0,1]
        
        mean_init_y = init_y.mean().item() # mean of y
        std_init_y = init_y.std().item()  # std of y
        
        norm_init_y = (init_y-mean_init_y)/std_init_y  # standardize y    
        norm_init_Y_var = torch.full_like(norm_init_y, self.y_variance) # y with noise    
        norm_best_init_y = norm_init_y.max().item() # best stardized y
        #print(norm_init_x,norm_init_y,norm_init_Y_var)
        
        single_model = FixedNoiseGP(norm_init_x, norm_init_y, norm_init_Y_var) # define GP: single task homoskedastic exact GP
    
        mml = ExactMarginalLogLikelihood(single_model.likelihood,single_model) # define likelihood to fit GP:The exact marginal log likelihood (MLL) for an exact Gaussian process with a Gaussian likelihood.

        fit_gpytorch_model(mml) # Fit hyperparameters of a GPyTorch model, L-BFGS-B via scipy.optimize.minimize().
        #fit_gpytorch_model(mml, optimizer=fit_gpytorch_torch) # Fit hyperparameters of a GPyTorch model, line search

        ###### define different acquisition functions
        #EI = ExpectedImprovement(model = single_model,best_f = norm_best_init_y) # define acquisition function by GP
        if ac_function == 'EI':
            EI = ExpectedImprovement(model = single_model,best_f = norm_best_init_y) # define acquisition function by GP
        elif ac_function == 'PI':
            EI = ProbabilityOfImprovement(model = single_model,best_f = norm_best_init_y)
        elif ac_function == 'UCB':
            EI = UpperConfidenceBound(model = single_model, beta = 0.2) # define acquisition function by GP
        elif ac_function == 'PM':
            EI = PosteriorMean(model= single_model)

        norm_candidates, _ = optimize_acqf(
                        acq_function = EI,
                        bounds = BO_bound,
                        q = BATCH_SIZE,
                        num_restarts = 50,
                        raw_samples = 1000)
            
        candidates = unnormalize(norm_candidates, bounds=bounds_init_x)

        return candidates.to(self.trip.ddtype)#, norm_candidates   # round to the lowest closet integer; change type back to 


    # define run rembo
    def REMBO_run_ac(self,BATCH_SIZE,N_BATCH,N_TRIALS,ac_function):
        """Runs the REMBO algorithm with a specified acquisition function across multiple trials.

        Args:
            BATCH_SIZE (int): Number of new candidate points per batch.
            N_BATCH (int): Number of optimization batches per trial.
            N_TRIALS (int): Number of independent trials to run.
            ac_function (str): Type of acquisition function to use ('EI', 'PI', 'UCB', 'PM').
        """
        global A_0, init_x_0, init_y_0, best_init_y_0, best_init_x_0 
        #print(A, init_x, init_y, best_init_y, best_init_x)
        warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        best_observed_all = []

        best_observed_all_x = []

        A_all = []

        # average over multiple trials
        for trial in range(1, N_TRIALS + 1):

            #A = gen_projection_rembo(low_dim,dim)

            A_all.append(A_0)


            print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
            best_observed = []
            best_observed_x = []


            # generate initial training data and initialize model
            init_x, init_y, best_init_y, best_init_x = init_x_0, init_y_0, best_init_y_0, best_init_x_0
            #best_observed_high = low_to_high_dimension(A,best_init_x)

            best_observed.append(best_init_y)
            best_observed_x.append(best_init_x)

            # run N_BATCH rounds of BayesOpt after the initial random batch
            for _ in range(1, N_BATCH + 1):    

                new_candidates = self.get_next_points_REMBO_ac(init_x, init_y, best_init_y, self.y_bound, BATCH_SIZE,ac_function)

                new_high = self.low_to_high_dimension(A_0,new_candidates)

                new_results = ChargingPlacement.Optimization_function(self, new_high).unsqueeze(-1)

                init_x = torch.cat([init_x,new_candidates])
                init_y = torch.cat([init_y,new_results])

                best_init_y = init_y.max().item()
                best_init_x = init_x[init_y.argmax().item()]

                #print(new_results)

                best_observed.append(best_init_y)
                best_observed_x.append(best_init_x)

            best_observed_all.append(best_observed)
            best_observed_all_x.append(best_observed_x)

        return best_observed_all,best_observed_all_x, init_x, init_y, A_all


    ########### add more trial
    def REMBO_add_trail(self, N_initial_,BATCH_SIZE_,N_BATCH_, N_TRIALS_, RUN_RESULT):
        """Adds more trials to an existing REMBO run.

        Args:
            N_initial_ (int): Number of initial data points for the new trials.
            BATCH_SIZE_ (int): Number of new candidate points per batch for the new trials.
            N_BATCH_ (int): Number of additional optimization batches per new trial.
            N_TRIALS_ (int): Number of new trials to run.
            RUN_RESULT (tuple): Existing results from a previous REMBO run.
        """
        warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        best_observed_all,best_observed_all_x, init_x, init_y, A_all = RUN_RESULT

        # average over multiple trials
        for trial in range(1, N_TRIALS_ + 1):

            A = self.gen_projection_rembo(self.low_dim, self.dim)

            A_all.append(A)


            print(f"\nTrial {trial:>2} of {N_TRIALS_} ", end="")
            best_observed = []
            best_observed_x = []


            # generate initial training data and initialize model
            init_x, init_y, best_init_y, best_init_x = self.generate_initial_data_REMBO(A, N_initial_)
            #best_observed_high = low_to_high_dimension(A,best_init_x)

            best_observed.append(best_init_y)
            best_observed_x.append(best_init_x)

            # run N_BATCH rounds of BayesOpt after the initial random batch
            for _ in range(1, N_BATCH_ + 1):    

                new_candidates = self.get_next_points_REMBO(init_x, init_y, best_init_y, self.y_bound, BATCH_SIZE_)

                new_high = self.low_to_high_dimension(A,new_candidates)

                new_results = ChargingPlacement.Optimization_function(self, new_high).unsqueeze(-1)

                init_x = torch.cat([init_x,new_candidates])
                init_y = torch.cat([init_y,new_results])

                best_init_y = init_y.max().item()
                best_init_x = init_x[init_y.argmax().item()]

                #print(new_results)

                best_observed.append(best_init_y)
                best_observed_x.append(best_init_x)

            best_observed_all.append(best_observed)
            best_observed_all_x.append(best_observed_x)
        
        return best_observed_all, best_observed_all_x, init_x, init_y, A_all


    ### RANDOM SEARCH ###

    # define collect initial points
    def generate_initial_data_BO(self, n):  # n is number of initial value want to generate
        """Generate initial data points for Bayesian Optimization using random search

        Args:
            n (int): number of initial values we want to generate
        """
        train_x = self.gen_high_dimension_variable(n)
        exact_obj = ChargingPlacement.Optimization_function(self, train_x).unsqueeze(-1).to(self.trip.ddtype)
        best_observation_value = exact_obj.max().item()
        best_observation_x = train_x[exact_obj.argmax().item()]

        return train_x,exact_obj, best_observation_value,  best_observation_x #train_x.float()


    # define run BO
    def Random_search(self, N_BATCH, N_TRIALS):
        """Performs a random search optimization strategy over multiple trials to find optimal settings.

        Args:
            N_BATCH (int): Number of batches of random searches to perform.
            N_TRIALS (int): Number of trials to execute.
        """   
        
        best_observed_all = []

        best_observed_all_x = []

        # average over multiple trials
        for trial in range(1, N_TRIALS + 1):

            print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
            best_observed = []
            best_observed_x = []

            rand_x = [np.random.uniform(self.x_real_l_bound.tolist(), self.x_real_u_bound.tolist()).astype(int)]
            init_x = torch.FloatTensor(rand_x).to(self.trip.ddtype)
            
            init_y = ChargingPlacement.Optimization_function(self, init_x).unsqueeze(-1).to(self.trip.ddtype)
            
            best_init_y = init_y.max().item()
            
            best_init_x = init_x[init_y.argmax().item()]
            
            best_observed.append(best_init_y)
            best_observed_x.append(best_init_x)

            # run N_BATCH rounds of BayesOpt after the initial random batch
            for _ in range(1, N_BATCH + 1):    
                
                rand_x_new = [np.random.uniform(self.x_real_l_bound.tolist(), self.x_real_u_bound.tolist()).astype(int)]
                init_x_new = torch.FloatTensor(rand_x_new).to(self.trip.ddtype)
            
                init_y_new = ChargingPlacement.Optimization_function(self, init_x_new).unsqueeze(-1).to(self.trip.ddtype)

        
                init_x = torch.cat([init_x,init_x_new])
                init_y = torch.cat([init_y,init_y_new])

                best_init_y = init_y.max().item()
                best_init_x = init_x[init_y.argmax().item()]

                best_observed.append(best_init_y)
                best_observed_x.append(best_init_x)

                print(".", end="")                    

            best_observed_all.append(best_observed)
            best_observed_all_x.append(best_observed_x)

        return best_observed_all,best_observed_all_x, init_x, init_y


    # add iteration
    def Random_add_iter(self, N_BATCH_, N_TRIALS_, RUN_RESULT):
        """Adds more iterations to an existing random search result to potentially improve the optimization outcome.

        Args:
            N_BATCH_ (int): Number of additional batches to run.
            N_TRIALS_ (int): Number of new trials to execute.
            RUN_RESULT (tuple): Contains the current best observations and their corresponding inputs.
        """
        best_observed_all = RUN_RESULT[0]
        best_observed_all_x = RUN_RESULT[1]
        init_x = RUN_RESULT[2]
        init_y = RUN_RESULT[3]

        # average over multiple trials
        for trial in range(1, N_TRIALS_ + 1):

            print(f"\nTrial {trial:>2} of {N_TRIALS_} ", end="")
            best_observed = best_observed_all[trial - 1]
            best_observed_x = best_observed_all_x [trial - 1]

            # run N_BATCH rounds of BayesOpt after the initial random batch
            for _ in range(1, N_BATCH_ + 1):    
                
                rand_x_new = [np.random.uniform(self.x_real_l_bound.tolist(), self.x_real_u_bound.tolist()).astype(int)]
                
                init_x_new = torch.FloatTensor(rand_x_new).to(self.trip.ddtype)
                #print(rand_x_new)
                init_y_new = ChargingPlacement.Optimization_function(self, init_x_new).unsqueeze(-1).to(self.trip.ddtype)
                #print(init_y_new)
        
                if init_y_new >= best_observed[-1]:
                    best_observed.append(init_y_new)
                    best_observed_x.append(init_x_new)
                else:
                    best_observed.append(best_observed[-1])
                    best_observed_x.append(best_observed_x[-1])
            
                print(".", end="")                    

        return best_observed_all, best_observed_all_x, init_x, init_y


