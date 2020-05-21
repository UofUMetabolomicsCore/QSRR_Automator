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

"""
this is designed to run the machine learning classes
this is in a separate module because the classes and their superclasses are quite large 
the imported module also has a couple functions needed to parse input functions (a couple need to load data for later use in the classes, so may as well put them all there) which will further lower the size of this module so this can just be infrastructure
it is also necessary to put in a graph  of the resutls, so this is a good place to put this 
"""
import pandas as pd
from numpy import geomspace
import os
import time    
from copy import deepcopy #not sure if we need deepcopy, but we need some form of copy and this will ensure no issues for any updates (and the time should be negligible compared to the time for the machine learning)

import joblib 
from sklearn.utils import shuffle

import generic_functions as gf
import machine_learning_rt_prediction as mlrp
import graph_model as gm
import load_check_previous_model_module as lcpm

NON_FEATURE_COLUMNS = 3
SVR_KERNEL = 'rbf'
saved_model_name = "saved_model.joblib"
saved_model_settings = "saved_model_settings.csv"
saved_model_features = "chosen_features.csv"
scores_used_in_model_generation = "cross_validation_and_final_sores.csv" 

def model_training_manager(input_data, infile, settings, programmer_settings):     
    """
    #need to print out the settings
    user_settings_to_write = pd.Series(settings.saved_settings, name = "Setting Value Used")
    programer_settings_to_write = pd.Series(programmer_settings.saved_settings, name = "Setting Value Used")
    user_settings_to_write.to_csv(os.path.join(settings.saved_settings["Output Folder"], "model training user settings.csv"), index = True, index_label = "Settings", header = True)
    programer_settings_to_write.to_csv(os.path.join(settings.saved_settings["Output Folder"], "model training programmer settings.csv"), index = True, index_label = "Settings", header = True)
    """
    original_settings = deepcopy(settings)
    number_of_features = len(input_data.columns) - NON_FEATURE_COLUMNS #remove rt, smiles and names
    
    #now we need to set the machine learning object
    why_are_eternal_loops_so_useful = True
    while(why_are_eternal_loops_so_useful): #if user wants to redo we can. most useful of coures if 
        settings_used_for_last_model = deepcopy(settings)
        start = time.time()
        variables_to_test = {} #this is for the param_grid in in the pipeline
        """
        #only uncomment if we need to have multiple possible values for manual (and adjust the calls to the subclasses and the superclass as well
        if settings["Feature Selection Method"] == "Manual":
            #since this is constant we could stick it into machine learning rt prediction, but this keeps the pipeline and we can easily expand the number of features easily if needed
            variables_to_test["feature_selection__max_features"] = settings["Target Number Features for Manual"]
            #variables_to_test["feature_selection__max_features"] = feature_creation(settings["Min Number of Features"], settings["Max Number of Features"], settings["Number of Feature Steps"])
        """    
        if settings["Model To Use"] == "Random Forest":
            Machine_Learning_Object = mlrp.Random_Forest_Subclass(settings["Number of Cross Validations"], settings["Processors_to_Use"], programmer_settings["Trees in Random Forest"], settings["Feature Selection Method"], programmer_settings["Auto Feauture Selection Step Size"], settings["Automatic Feature Selection CV"], settings["Min Number of Features"],settings["Target Number Features for Manual"])
        elif settings["Model To Use"] == "SVR":
            #need to use a log space differentce, geomspallows min and max setting quite easily
            variables_to_test["machine_learning_step__C"] = geomspace(settings["C min"], settings["C max"], settings["C number of steps"])
            variables_to_test["machine_learning_step__gamma"] = geomspace(settings["gamma min"], settings["gamma max"], settings["gamma number of steps"])
            Machine_Learning_Object =mlrp.SVR_Subclass(SVR_KERNEL, variables_to_test, settings["Number of Cross Validations"], settings["Processors_to_Use"], programmer_settings["Trees in Random Forest"], settings["Feature Selection Method"], programmer_settings["Auto Feauture Selection Step Size"], settings["Automatic Feature Selection CV"], settings["Min Number of Features"], settings["Target Number Features for Manual"])
        
        elif settings["Model To Use"] == "LR":
            Machine_Learning_Object = mlrp.LR_Subclass(settings["Number of Cross Validations"], settings["Processors_to_Use"], programmer_settings["Trees in Random Forest"], settings["Feature Selection Method"], programmer_settings["Auto Feauture Selection Step Size"], settings["Automatic Feature Selection CV"], settings["Min Number of Features"],settings["Target Number Features for Manual"])
        elif settings["Model To Use"] == "Choose Best":
            svr_variables = {"machine_learning_step__C": geomspace(settings["C min"], settings["C max"], settings["C number of steps"]), "machine_learning_step__gamma": geomspace(settings["gamma min"], settings["gamma max"], settings["gamma number of steps"])}
            
            Machine_Learning_Object =mlrp.Choice_Subclass(SVR_KERNEL, svr_variables, settings["Number of Cross Validations"], settings["Processors_to_Use"], programmer_settings["Trees in Random Forest"], settings["Feature Selection Method"], programmer_settings["Auto Feauture Selection Step Size"], settings["Automatic Feature Selection CV"], settings["Min Number of Features"], settings["Target Number Features for Manual"])
        
        #objects are created, now let's feed them data
        #get the input data 
        #before we can actually do the analysis we need to shuffle the data for testing and to prevent user order from messing up cross validations (it won't autoshuffle)
        simple_shuffle = shuffle(input_data)
        Machine_Learning_Object.format_input_data(simple_shuffle) 
        Machine_Learning_Object.actual_testing_calculation()
        print ("Time taken: {} min".format((time.time()-start)/60))
        #now we have all the data. we need a function to graph the data and ask the user if they wish to redo (if they do, make sure to randomize)
        #show_model(Machine_Learning_Object)
        display_object = gm.display_training_results(Machine_Learning_Object, settings["Output Folder"], settings, settings_used_for_last_model, original_settings, number_of_features)
        display_object.exec_()
        if display_object.return_value == "Not Selected":
            return False, "", "", "" 
        elif display_object.return_value == "Accept Model":
            #if choose best is used we need to know what the actual choice was.  we will make it a string for easier comparison
            if settings["Model To Use"] == "Choose Best":
                #print (type(Machine_Learning_Object.best_estimator['machine_learning_step']))
                #print (str(Machine_Learning_Object.best_estimator['machine_learning_step']))
                ml_object_string = str(Machine_Learning_Object.best_estimator['machine_learning_step'])
                best_choice_string = ml_object_string[:ml_object_string.index("(")]
            else:
                best_choice_string = "" 
            #used to save the data we will need for the 
            series_to_save = pd.Series({"Training File": infile,  "Final Model Score": Machine_Learning_Object.final_r2_score,  "Feature Selection Method": settings["Feature Selection Method"]})
            #these are only for cases where the software was choosing.  if it didn't make a choice, say no test for number of feature values, they should be able to determine things from the saved settings
            #the ifs are a bit complicated since choose best is also valid.  if this is expanded 
            if (settings["Model To Use"] == "SVR" or best_choice_string == "SVR") and settings["C number of steps"] != 1: 
                series_to_save["C Value"] = Machine_Learning_Object.best_estimator['machine_learning_step__C']
            elif settings["Model To Use"] == "SVR" or best_choice_string == "SVR": #may as well report what the C is 
                series_to_save["C Value"] = settings["C min"]
            if (settings["Model To Use"] == "SVR" or best_choice_string == "SVR") and settings["gamma number of steps"] != 1: 
                series_to_save["gamma Value"] = Machine_Learning_Object.best_estimator['machine_learning_step__gamma']
            elif settings["Model To Use"] == "SVR" or best_choice_string == "SVR": #may as well report what the C is 
                series_to_save["gamma Value"] = settings["gamma min"]
            
            #since we are choosing which model to use, we should actually save this.  not necessary for loading the model, but it is good to have 
            if settings["Model To Use"] == "Choose Best":
                series_to_save["Machine Learning Model"] = best_choice_string
            else:
                series_to_save["Machine Learning Model"] = settings["Model To Use"]
            series_to_save.index.name = "Setting"
            series_to_save.name = "Value"
            
            #we need to deal with saving cross validation and final scores
            #we don't need to bother with gridsearchcv results or anything since that would be confusing to the user (to any actual programmer dealing with this can just adjust the code and should know enough to interpret the results)
            #we'll start with final results, then do the cross validations in order of the value used 
            
            score_series = pd.Series({"Final Model R2 Score": Machine_Learning_Object.final_r2_score, 
            "Final Model Mean Absolute Error": Machine_Learning_Object.final_mae, 
            "Cross Validation R2 Values": Machine_Learning_Object.r2_data, 
            "Mean of Cross Validation R2 Values": Machine_Learning_Object.r2_mean,
            "Median of Cross Validation R2 Values": Machine_Learning_Object.r2_median,
            "Standard Deviation of Cross Validation R2 Values": Machine_Learning_Object.r2_std, 
            "Cross Validation Mean Absolute Error Values": Machine_Learning_Object.mean_absolute_error,
            "Mean of Cross Validation Mean Absolute Error Values": Machine_Learning_Object.mae_mean,
            "Median of Cross Validation Mean Absolute Error Values": Machine_Learning_Object.mae_median,
            "Standard Deviation of Cross Validation Mean Absolute Error Values": Machine_Learning_Object.mae_std})
            score_series.index.name = "Scoring Metric"
            score_series.name = "Value(s)"
            
            return True, Machine_Learning_Object, series_to_save, score_series #this is the object we need to perform future analyses


#we need to save the data to a file for later use.
#this may  need to move to the main when we sort out how to ask the user if they wish to continue
#only use .csv (this is a file purely for use by the software so this shouldn't be an issue)
def save_data(ml_object, settings, series_to_save, score_series, excel_object = None):
    saving_csv_names = [saved_model_settings, scores_used_in_model_generation]
    list_of_series_to_save =[series_to_save, score_series]
    for i in range(len(list_of_series_to_save)) :
        stuff = gf.write_single_file(os.path.join(settings["Output Folder"],saving_csv_names[i]), list_of_series_to_save[i], True, False) #we'll write the settings as .csvs always to ensure that we can easily access as needed and not have to search for them (if they want excel we'll shove it into a sheet on the main excel sheet as well)
        if stuff != "Success":
            return stuff
    if excel_object:
        series_to_save.to_excel(excel_object, "Model Settings")
        score_series.to_excel(excel_object, "Model Score Values")
    #this comes from https://scikit-learn.org/stable/modules/model_persistence.html accessed 
    joblib.dump(ml_object.main_pipeline, os.path.join(settings["Output Folder"], saved_model_name))
    if settings["Feature Selection Method"] != "None":
        feature_series = pd.Series(ml_object.chosen_features, name = "Chosen Features")
        stuff = gf.write_single_file(os.path.join(settings["Output Folder"],saved_model_features), feature_series, False, False)
        if stuff != "Success":
            return stuff
        if excel_object:
            feature_series.to_excel(excel_object, "Selected Features", index = False)
    return stuff
    
#may also need to move this.  needs to load stuff, and check if everything is good
"""checks needed:
-check both files exist
-is file still valid
-run the check
-confirm that the score is within acceptable range
"""  

def load_data(output_folder, programmer_settings):
    if os.path.isfile(os.path.join(output_folder,saved_model_features)):
        features_file = os.path.join(output_folder,saved_model_features)   
    else:
        features_file = None
    unknown_predictor_object = lcpm.load_check_previous_model(os.path.join(output_folder,saved_model_name),os.path.join(output_folder, saved_model_settings), programmer_settings["Allowed Model Loading Error"],features_file)  
    #we're going to assume the user isn't stupid enough to modify these files for no reason
    if unknown_predictor_object.error == "":
        return unknown_predictor_object, unknown_predictor_object.all_descriptors, unknown_predictor_object.feature_selection_method
    else:
        return unknown_predictor_object.error, "", ""
        
def analyze_user_data(all_data, model_object, settings, features_needed = None):
    all_data = all_data.set_index("Compound Name")
    all_descriptors = all_data.drop("SMILES", axis = 1)
    #need to predict from all descriptors
    #first, if we have features needed, we need to be sure that those features are there
    if type(features_needed) != type(None):
        try:
            all_descriptors[features_needed]
        except KeyError:
            return "Features for RT prediction do not match those used in the model.  Please correct and try again."
    try:
        prediction = model_object.predict(all_descriptors)
    except ValueError:
        return "Prediction failed likely because the amount of features in the model and in file for prediction are different."
    return prediction
    #then assign the RT to the compounds and smiles (leave off the descriptors for this output)
    #also need to check that the descriptors are the same use the try except like in lcpm
    