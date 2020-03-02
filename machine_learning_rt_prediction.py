"""
Copyright (c) 2020 Bradley Naylor, James Cox and University of Utah

All rights reserved.

Redistribution and use in source and binary forms,
with or without modification, are permitted provided
that the following conditions are met:

    * Redistributions of source code must retain the
      above copyright notice, this list of conditions
      and the following disclaimer.
    * Redistributions in binary form must reproduce
      the above copyright notice, this list of conditions
      and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the author nor the names of any contributors
      may be used to endorse or promote products derived
      from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
#correctly running the analysis 
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline 
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error 
from sklearn.linear_model import LinearRegression
import numpy as np
#this super class should have all necessary caluclations
#it should open the file, make output, and run all calculations
#the subclasses should create the machine learning object and set if we are scaling the data or not

"""
we need the following functions:
    xinit - always need the init class.  can define local variables here.  the subclasses can shift as necessary
    -let's feed in a filename we have already checked with general functions 
    -check the input form briefly
    -essentially copy gc_rt_with_correct_protocol
    -get the range from settings
    -make a "variables" class so we can fiddle with things later (change number of variables, access as necessary)
    -starting scores and step sizes from settings
    -any constant variables (like a huge number of trees for random forest) can also be in settings
    -number of tests in the settings (do not let this be set under 100)
    -need a function to run the test set to make the object and report the results
    -this should include a graph of how the test set looked on the data and how the training set worked
    -if the user confirms (can run the graph as a seperate function in this module and ask the user in the main) then do any actual data they have
    -save graphs to pdfs if confirmed and for results of the test
    -see if we can link to a progress bar of some to see if we have stalled during model creation (fitting data should be fast enough)
    -param grid needs to be in the form of {"step_name__variable_name": [], }the list must contain all variables you wish to check so [1,2,3] will run the variable as 1, 2, and 3.
        note that each variable must be tried with each other variables as many times as the cv_parameter says.  therefore each extra element causes an exponential increase in running time
        try to limit the number of different variables in the list. (though set as many as you need to.  setting a 1 value doesn't slow anything unless the effect fo that value slows the function in some way)
        
"""

"""
IMportant notes:
currently we are doing multiple tests to determine the best.  feature selection and prediction are separate which is going to be problematic.
Since pipeline is built to do just this, it should be used to do this.
this requires a few things:
        xneed to add the feature selection to the pipline
        -check the test set without performing a separate selection on that (should do automatically but should check)
        -adjust choose self.choose_best_feature to work with the chosen features for a few (choose a number from settings) to choose best features
        -add a true/false check (or separate function with the call in the main)to avoid feature selection if necessary
        -keep cv_numbers small and maybe randomize numbers instead of using range?
        -in the main make a setting for void volume, trimming all before that poing
looking at things we can use RFECV to use a covariance matrix, choose the best number of features and what they are and then get them out of the object
"""


     
               

#$do not call this class on it's own.  it exists solely to be a parent class for other classes it will not function properly on its own.
class Machine_Learning_Superclass(object):
    def __init__(self, Number_of_Cross_Validations, Processors_to_Use, Trees_in_Forest, Feature_Selection_Method, auto_feature_selection_step_size, auto_feature_selection_cv, minimum_feature_allowed_for_auto, manual_features_desired): #feed settings in as **kwargs (so **settings)
        self.cross_validations = Number_of_Cross_Validations
        self.number_of_processors_to_use = Processors_to_Use
        self.size_of_forest = Trees_in_Forest
        self.feature_selection = Feature_Selection_Method
        self.auto_step_size = auto_feature_selection_step_size
        self.auto_cv = auto_feature_selection_cv
        self.auto_min_features = minimum_feature_allowed_for_auto
        self.manual_features_desired = manual_features_desired
        self.feature_selection_option() #for now all subclasses use the same options.  can be shifted to the subclasses if necessary
    
    #a placeholder class that serves as a reminder to actually use the child classes
    def make_pipeline(self):
        raise NotImplementedError("cannot determine the machine learning type in the superclass")
    
    #for right now feature selection is based on Random Forest we can stick other options here if necessary.
    #(this function as a whole is unnecessary currently but provides a good place for assignment and easy modification)
    def feature_selection_option(self):
        if self.feature_selection == "Manual":
            #the selectfrom model attempts to set a threshold and keeps all features above that  (actually a pretty good way of auto feature selection) however if threshold is -np.inf the max_number argument sets the number of features chosen
            #for now we'll define the max_number here so it will be easier to use, but this can be adjusted later if we need to
            self.actual_feature_selection_function = SelectFromModel(RandomForestRegressor(n_estimators = self.size_of_forest), threshold = -np.inf, max_features = self.manual_features_desired)
        elif self.feature_selection == "Automatic":
            self.actual_feature_selection_function = RFECV(RandomForestRegressor(n_estimators = self.size_of_forest), cv = self.auto_cv, step = self.auto_step_size, min_features_to_select = self.auto_min_features)  
                                                                    
    def format_input_data(self, correct_form):
        correct_form = correct_form.set_index("Compound Name")
        correct_form = correct_form.drop("SMILES", axis = 1)
        self.analysis_descriptors = list(correct_form.columns)
        self.analysis_descriptors.remove("RT")
        self.y_values = correct_form["RT"].astype(float)
        self.x_values = correct_form[self.analysis_descriptors].astype(float)       
    
    #this could be merged with actual_testing_calculation. this function is largely for organizational purposes
    def calculate_report_variables(self):
        #we're going to start with metrics on final model
        self.predicted_y = self.main_pipeline.predict(self.x_values)
        self.final_r2_score = self.main_pipeline.score(self.x_values, self.y_values)
        self.final_mae = mean_absolute_error(self.y_values, self.predicted_y)
        
        
        #need to determine what the best parameters are if such were calculated
        if self.grid_search:
            self.best_estimator = self.main_pipeline.named_steps["gridsearchcv_step"].best_params_
            
        #need to get the features if these were calculated (don't need to bother with none obviously)
        if self.feature_selection == "Automatic":
            mask = self.main_pipeline.named_steps["feature_selection_step"].support_
            self.chosen_features = self.x_values.columns[mask]
        elif self.feature_selection == "Manual":
            if self.grid_search:
                mask = self.main_pipeline.named_steps["gridsearchcv_step"].best_estimator_.named_steps["feature_selection_step"].get_support()
            else: #for right now only the random forest subclass will use this but more might be needed in the future
                mask = self.main_pipeline.named_steps["feature_selection_step"].get_support()
            self.chosen_features = self.x_values.columns[mask]
            
        self.all_required_features = list(self.x_values.columns)
        #now we need to get the cross_validation outputs
        #will only bother with the test results, no need to bother with training
        #we'll start with r2 
        self.r2_data = self.cross_validation_scores['test_r2_analysis']
        self.r2_median, self.r2_mean, self.r2_std = np.median(self.r2_data), self.r2_data.mean(), self.r2_data.std()
        #now we need to do the actual error (it starts negative and we need to get positive to make any sense)
        self.mean_absolute_error = [-x for x in self.cross_validation_scores["test_time_error_analysis"]]
        self.mae_median, self.mae_mean, self.mae_std = np.median(self.mean_absolute_error), np.mean(self.mean_absolute_error), np.std(self.mean_absolute_error)
        
    
    #now we need to actually do the analysis.
    #the main metrics we are going to be using are r2 and mean absolute error since these will be most understandable to non-statisticians
    def actual_testing_calculation(self):        
        self.make_pipeline()
        #we are going to use the cross_validate function instead of cross_val_scores to use multiple metrics at once
        scores_needed = {"r2_analysis": 'r2', "time_error_analysis": 'neg_mean_absolute_error'}
        #do the actual cross_validation.  
        self.cross_validation_scores = cross_validate(self.main_pipeline, self.x_values, self.y_values, cv = self.cross_validations, n_jobs = self.number_of_processors_to_use, verbose = 2, scoring = scores_needed) 
        self.main_pipeline.fit(self.x_values, self.y_values)
        self.calculate_report_variables()
        


#$if adding a new class, keep the variable names the same.  including the "scaler_step" and "machine_learning_step" as the pipeline portion names
class Random_Forest_Subclass(Machine_Learning_Superclass):
    def __init__(self, Number_of_Cross_Validations, Processors_to_Use, Trees_in_Forest, Feature_Selection_Method, auto_feature_selection_step_size, auto_feature_selection_cv, minimum_feature_allowed_for_auto, manual_features_desired):
        Machine_Learning_Superclass.__init__(self, Number_of_Cross_Validations, Processors_to_Use, Trees_in_Forest, Feature_Selection_Method, auto_feature_selection_step_size, auto_feature_selection_cv, minimum_feature_allowed_for_auto, manual_features_desired) #get the basic values (can reset later if need be)
        #currently we are assuming a large number of trees good and random state is irrelevant to the final results (or at least the effect is sufficiently random that testing different settings is a waste of time)
        #as a result the cross validation should get sufficient results to see how well random forest as a model works so for now there is nothing to use a grid search on
        self.grid_search = False

    def make_pipeline(self):
        #the pipeline will eventually need to be a pipeline not a list. starting with a list makes it easier to add things and make adjustments 
        #no second pipeline in this case
        self.main_pipeline = [] #main pipeline to feed into the cross validation and use to create the final model
        if self.feature_selection == "Manual" or self.feature_selection == "Automatic":
            #doesn't actually cost too much and doesn't have a cv like the rfecv so we'll put in a gridsearch cv (important for the final model)
            self.main_pipeline.append(("feature_selection_step", self.actual_feature_selection_function))
        #there is no point in putting a gridsearchcv if we have no parameters to select (neither manual features or parameters of the estimator) it takes time and computing resources for no reason
        self.main_pipeline.append(("machine_learning_step", RandomForestRegressor(n_estimators = self.size_of_forest)))
            
        self.main_pipeline = Pipeline(self.main_pipeline) #need a pipeline not a list
       
class SVR_Subclass(Machine_Learning_Superclass):
    def __init__ (self, kernel, settings_to_check, Number_of_Cross_Validations, Processors_to_Use, Trees_in_Forest, Feature_Selection_Method, auto_feature_selection_step_size, auto_feature_selection_cv, minimum_feature_allowed_for_auto, manual_features_desired):
        Machine_Learning_Superclass.__init__(self, Number_of_Cross_Validations, Processors_to_Use, Trees_in_Forest, Feature_Selection_Method, auto_feature_selection_step_size, auto_feature_selection_cv, minimum_feature_allowed_for_auto, manual_features_desired) #get the basic values (can reset later if need be)
        self.kernel = kernel 
        self.all_grid_search_variables = settings_to_check
         
    def make_pipeline(self):
        #not sure if scaling is most appropriately done here or in the gridsearch cv, but we may as well just start with it  since it needs to be done no matter what and contains no random elements
        self.main_pipeline = [("scaler_step", MinMaxScaler())] #main pipeline to feed into the cross validation and use to create the final model
        self.secondary_pipeline = [] #this is for a pipeline for any gridsearchcv inside the main pipeline
        if self.feature_selection == "Manual":
            self.secondary_pipeline.append(("feature_selection_step", self.actual_feature_selection_function))
        elif self.feature_selection == "Automatic":
            self.main_pipeline.append(("feature_selection_step", self.actual_feature_selection_function))
            
        if len(self.all_grid_search_variables["machine_learning_step__C"]) >1 or len(self.all_grid_search_variables["machine_learning_step__gamma"]) >1 or self.secondary_pipeline != []:
            self.secondary_pipeline.append(("machine_learning_step", SVR(kernel = self.kernel)))
            self.main_pipeline.append(("gridsearchcv_step", GridSearchCV(Pipeline(self.secondary_pipeline), param_grid = self.all_grid_search_variables, cv = self.cross_validations)))#, iid = False)))
            self.grid_search = True
        else:
            self.main_pipeline.append(("machine_learning_step", SVR(kernel = self.kernel, C = self.all_grid_search_variables["machine_learning_step__C"][0], gamma = self.all_grid_search_variables["machine_learning_step__gamma"][0]))) 
            self.grid_search = False
        self.main_pipeline = Pipeline(self.main_pipeline)

#not sure how well this will work out, but it is important for testing and it hurts nothing
#this will mostly be based on the Random Forest class because there is no need for grid search (no relevant variables)
#this should not be done without feature selection (it's going to overfit because it can't prune out features)
class MLR_Subclass(Machine_Learning_Superclass):
    def __init__(self, Number_of_Cross_Validations, Processors_to_Use, Trees_in_Forest, Feature_Selection_Method, auto_feature_selection_step_size, auto_feature_selection_cv, minimum_feature_allowed_for_auto, manual_features_desired):
        Machine_Learning_Superclass.__init__(self, Number_of_Cross_Validations, Processors_to_Use, Trees_in_Forest, Feature_Selection_Method, auto_feature_selection_step_size, auto_feature_selection_cv, minimum_feature_allowed_for_auto, manual_features_desired) #get the basic values (can reset later if need be)               
        self.grid_search = False # no need to actually do a grid search here (only variable that is really relevant
    def make_pipeline(self):
        #needs a scaler to be useful and this is more consistent with our other classes than using the internal normalize function
        self.main_pipeline = [("scaler_step", MinMaxScaler())] #main pipeline to feed into the cross validation and use to create the final model
        if self.feature_selection == "Manual" or self.feature_selection == "Automatic":
            #doesn't actually cost too much and doesn't have a cv like the rfecv so we'll put in a gridsearch cv (important for the final model)
            self.main_pipeline.append(("feature_selection_step", self.actual_feature_selection_function))
        #there is no point in putting a gridsearchcv if we have no parameters to select (neither manual features or parameters of the estimator) it takes time and computing resources for no reason
        self.main_pipeline.append(("machine_learning_step", LinearRegression()))
            
        self.main_pipeline = Pipeline(self.main_pipeline) #need a pipeline not a list
            

#this subclass deals with the possibility of having the software choose the algorithm that is best for this dataset (and therefore the user doesn't have to at the cost of some calculation time)
#remember to adjust if amy more algorithms are added
class Choice_Subclass(Machine_Learning_Superclass):
    def __init__(self, kernel, svr_variables, Number_of_Cross_Validations, Processors_to_Use, Trees_in_Forest, Feature_Selection_Method, auto_feature_selection_step_size, auto_feature_selection_cv, minimum_feature_allowed_for_auto, manual_features_desired):
        Machine_Learning_Superclass.__init__(self, Number_of_Cross_Validations, Processors_to_Use, Trees_in_Forest, Feature_Selection_Method, auto_feature_selection_step_size, auto_feature_selection_cv, minimum_feature_allowed_for_auto, manual_features_desired)
        self.kernel = kernel 
        self.svr_variables = svr_variables
        self.grid_search = True#no point checking if we need a gridsearch since it will be required to determine which method is best
        
    def make_pipeline(self):
        self.all_grid_search_variables = []
        #for this we will make the machine lerning step and then merge with the already provided variables.  add all to the all_grid_search_variables list
        rf_dict = {"machine_learning_step": [RandomForestRegressor(n_estimators = self.size_of_forest)]}
        self.all_grid_search_variables.append(rf_dict)
        
        svr_dict = {"machine_learning_step": [SVR(kernel = self.kernel)]}
        for key in self.svr_variables.keys(): svr_dict[key] = self.svr_variables[key]
        self.all_grid_search_variables.append(svr_dict)
        
        mlr_dict = {"machine_learning_step": [LinearRegression()]}
        self.all_grid_search_variables.append(mlr_dict)
        
        
        #pipelines like the other subclasses
        #since we are choosing between multiple models that either need scaling (SVR) or don't care about scaling (Random Forest) we may as well start with scaling since it can't hurt anything
        self.main_pipeline = [("scaler_step", MinMaxScaler())] #main pipeline to feed into the cross validation and use to create the final model
        self.secondary_pipeline = [] #need to be filled after 
        #don't forget feature selection
        if self.feature_selection == "Manual":
            self.secondary_pipeline.append(("feature_selection_step", self.actual_feature_selection_function))
        elif self.feature_selection == "Automatic":
            self.main_pipeline.append(("feature_selection_step", self.actual_feature_selection_function))
        
        self.secondary_pipeline.append(("machine_learning_step", RandomForestRegressor())) #this is just a placeholder for the step in the pipeline.  the actual function in here is irrelevant and will not be used
        self.main_pipeline.append(("gridsearchcv_step", GridSearchCV(Pipeline(self.secondary_pipeline), param_grid = self.all_grid_search_variables, cv = self.cross_validations)))#, iid = False)))
        self.main_pipeline = Pipeline(self.main_pipeline)