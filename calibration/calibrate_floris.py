import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from floris import FlorisModel
import yaml


def setup_floris_yaml(input_file,setup_params,output_file = None):
    # Load the existing YAML file
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)

    def set_nested_value(d, keys, value):
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        if value is None:
            # Remove the key if value is None
            d.pop(keys[-1], None)
        else:
            d[keys[-1]] = value
    # Apply modifications
    for path, value in setup_params.items():
        keys = path.split('.')
        set_nested_value(data, keys, value)

    if output_file == None: output_file=input_file

    # Write the modified data back to a new YAML file
    with open(output_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    return

class FLORIS_Optimizer:

    def __init__(self,floris_models,calibration_params,target_data,calibration_bounds=None,turbine_calibration_params = None,turbine_calibration_bounds = None):

        self.floris_models = floris_models
        self.calibration_params = calibration_params
        self.turbine_calibration_params = turbine_calibration_params
        self.cases = floris_models.keys()
        self.num_cases = len(self.cases)
        self.target_data_dict = target_data 
        self.get_target_data()
        
        if self.turbine_calibration_params != None:
            self.turbine_files = {}
            for case_iter, case in enumerate(self.cases):
                floris_model = self.floris_models[case]
                floris_model_farm = floris_model.core.farm
                self.turbine_files[case] = {}
                for t in floris_model_farm.turbine_type:
                    self.turbine_files[case][t] = None
                    if t in self.turbine_calibration_params.keys():
                        internal_fn = (floris_model_farm.internal_turbine_library / t).with_suffix(".yaml")
                        external_fn = (floris_model_farm.turbine_library_path / t).with_suffix(".yaml")
                        in_internal = internal_fn.exists()
                        in_external = external_fn.exists()
                        if in_internal:
                            full_path = internal_fn
                        elif in_external:
                            full_path = external_fn
                        self.turbine_files[case][t] = full_path

        self.x0 = np.asarray(self.calibration_dict_to_array(calibration_params))
        self.num_params = []
        if calibration_bounds == None:
            self.bounds = []
            for i in range(len(self.x0)):
                self.bounds.append((None,None))
        else:
            self.bounds = self.calibration_dict_to_array(calibration_bounds)
        self.num_params.append(len(self.x0)) #number of wake parameters
        
        if self.turbine_calibration_params != None:
            for titer, t in enumerate(self.turbine_calibration_params.keys()):
                x0 = np.asarray(self.calibration_dict_to_array(self.turbine_calibration_params[t]))
                self.num_params.append(len(x0) + self.num_params[-1])
                self.x0 = np.concatenate((self.x0, x0))

                if turbine_calibration_bounds == None or turbine_calibration_bounds[t] == None:
                    for i in range(len(self.x0)-self.num_params[0]):
                        self.bounds.append((None,None))
                else:
                    self.bounds += self.calibration_dict_to_array(turbine_calibration_bounds[t])

        return

    def optimize(self,maxiter=1000):
        MLE = sp.optimize.minimize(
            self.cost_function,
            self.x0,
            bounds=self.bounds,
            options={"disp": True, "maxiter": maxiter},
        )
        return MLE

    def cost_function(self,x):
        self.calibration_params = self.calibration_array_to_dict(self.calibration_params,x[0:self.num_params[0]])
        if self.turbine_calibration_params != None:
            for titer, t in enumerate(self.turbine_calibration_params.keys()):
                self.turbine_calibration_params[t] = self.calibration_array_to_dict(self.turbine_calibration_params[t],x[self.num_params[titer]:self.num_params[titer+1]])
        #try:
        obs = self.run_floris_models()
        llhood = 0
        scaling = 1.0/1000.0
        for case_iter , case in enumerate(self.cases):
            llhood += 0.5 * np.sum((obs[case]*scaling - self.target_data[case]) ** 2)
        #except Exception:
        #    llhood = -np.inf
        return llhood

    def get_target_data(self):
        self.target_data = {}
        for case in self.cases:
            file = self.target_data_dict[case][1]
            floris_model = self.floris_models[case]
            target_data = np.zeros((1,floris_model.n_turbines))
            df = np.asarray(pd.read_csv(file, header=None))
            self.target_data[case] = df
        return

    def run_floris_models(self):
        results = {}
        for case_iter, case in enumerate(self.cases):
            floris_model = self.floris_models[case]
            self.modify_yaml(floris_model.configuration,self.calibration_params)
            if self.turbine_calibration_params != None:
                for t in self.turbine_calibration_params.keys():
                    self.modify_yaml(self.turbine_files[case][t],self.turbine_calibration_params[t])

            self.floris_models[case] = FlorisModel(floris_model.configuration)
            self.floris_models[case].run()
            qoi = self.target_data_dict[case][0]
            if qoi == 'turbine_power':
                results[case] = self.floris_models[case].get_turbine_powers()
        return results

    def modify_yaml(self,input_file,params,output_file = None):
        # Load the existing YAML file
        with open(input_file, 'r') as file:
            data = yaml.safe_load(file)

        def set_nested_value(d, keys, value):
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            if value is None:
                # Remove the key if value is None
                d.pop(keys[-1], None)
            else:
                d[keys[-1]] = value
        # Apply modifications
        for path, value in params.items():
            keys = path.split('.')
            set_nested_value(data, keys, value)

        if output_file == None: output_file=input_file
        # Write the modified data back to a new YAML file
        with open(output_file, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        return

    def calibration_dict_to_array(self,params):
        params_vec = []
        for key, value in params.items():
            if isinstance(value, list):
                params_vec.extend(value)  # Add all elements from the list
            elif isinstance(value, (int, float, tuple)):
                params_vec.append(value)  # Add the single number
            elif value == None:
                params_vec.append(value)
        #params_vec = np.asarray(params_vec)
        return params_vec

    def calibration_array_to_dict(self,params,values):
        counter = 0
        for key, value in params.items():
            if isinstance(value, list):
                for i in range(len(value)): 
                    params[key][i] = float(values[counter])
                    counter += 1
            elif isinstance(value, (int, float)):
                params[key] = float(values[counter])
                counter += 1
        return params



#should we use inheretice to define a general reduce order model and then we can have a floris and rans option?


if __name__ == "__main__":

    cases = ["Baseline_3x3_6D_wind_farm",]

    #3x3 wind farm positions with 6D spacing
    T00_base_position               = [1963.53247, 4000.0, 0.0]
    T01_base_position               = [2981.766235, 2981.766235, 0.0]
    T02_base_position               = [4000.0, 1963.53247, 0.0]
    T03_base_position               = [2981.766235, 5018.233765, 0.0]
    T04_base_position               = [4000.0, 4000.0, 0.0]
    T05_base_position               = [5018.233765, 2981.766235, 0.0]
    T06_base_position               = [4000.0 ,6036.46753, 0.0]
    T07_base_position               = [5018.233765, 5018.233765, 0.0]
    T08_base_position               = [6036.46753, 4000.0, 0.0]

    layout_x  = [T00_base_position[0],T01_base_position[0],T02_base_position[0],T03_base_position[0],T04_base_position[0],T05_base_position[0],T06_base_position[0],T07_base_position[0],T08_base_position[0]]
    layout_y  = [T00_base_position[1],T01_base_position[1],T02_base_position[1],T03_base_position[1],T04_base_position[1],T05_base_position[1],T06_base_position[1],T07_base_position[1],T08_base_position[1]]

    setup_params  = {
        'flow_field.wind_speeds': [9.0,],
        'flow_field.wind_directions': [225,],
        'flow_field.turbulence_intensities': [0.0309,],
        'flow_field.wind_veer': 8.94,
        'flow_field.wind_shear': 0.16,
        'farm.layout_x': layout_x,
        'farm.layout_y': layout_y
    }
    setup_files = {}
    setup_files[cases[0]] = "baseline_emgauss.yaml"

    #create list of floris models for each case to optimize
    floris_models = {}
    for case in cases:
        setup_floris_yaml(setup_files[case],setup_params)
        floris_model = FlorisModel(setup_files[case])
        floris_models[case] = floris_model
        print("Setup floris model for " + case + " with configuration file: ", floris_model.configuration)

    #initial values for calibration parameters
    calibration_params = {
        'wake.wake_velocity_parameters.empirical_gauss.wake_expansion_rates': [0.023,0.008],
    }

    calibration_bounds = {
        'wake.wake_velocity_parameters.empirical_gauss.wake_expansion_rates': [(0.0,1.0),(0.0,1.0)],
    }

    #example of including turbine parameters in calibraiton
    turbine_calibration_params= {}
    turbine_calibration_params['iea_15MW'] = {
        'TSR': 8,
    }

    turbine_calibration_bounds= {}
    turbine_calibration_bounds['iea_15MW'] = {
        'TSR': (0,20),
    }

    target_data = {}
    target_data[cases[0]] = ('turbine_power','./LES_data/MedWS_LowTI_Baseline_6D_45.csv')

    solver = FLORIS_Optimizer(floris_models,calibration_params,target_data,calibration_bounds=calibration_bounds)
    MLE = solver.optimize()
