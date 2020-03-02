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

#this is the class to handle actually predicting the retention times of unknonwns
import pandas as pd
#import sklearn.pipeline

import joblib 

class load_check_previous_model(object):
    def __init__(self, machine_learning_object_file, settings_file_loc, allowed_difference, chosen_features_file):
        self.error = "" #this is going to be the main reporter of errors.  early returns will alter this so we can check it for every time it may be altered
        self.allowed_diffence_in_score = allowed_difference
        try:
            self.ml_object=joblib.load(machine_learning_object_file)
        except:
            #functional
            self.error = "Unable to open {}. may have been corrupted or deleted.  Please correct".format(machine_learning_object_file)  
        if self.error == "":
            self.set_training_values(settings_file_loc)
        if self.error == "":#if error from the above function was set no need to do anything else
            self.check_training_file(chosen_features_file)
    
    def set_training_values(self, settings_file_loc):
        try:
            settings = pd.read_csv(settings_file_loc, index_col = "Setting")
            self.training_file = settings["Value"]["Training File"]
            self.training_score = float(settings["Value"]["Final Model Score"])
            self.feature_selection_method = str(settings["Value"]["Feature Selection Method"])
        except:
            #functional
            self.error = "{} does not exist or has been altered.  Please correct.".format(settings_file_loc)
        
    def check_training_file(self, chosen_features_file):
        try:
            if self.training_file[-4:] == ".csv":
                training_data = pd.read_csv(self.training_file)
            elif self.training_file[-5:] == ".xlsx":
                training_data = pd.read_excel(self.training_file)
            else:
                #functional
                self.error = "{} is not a .csv or .xlsx. settings file has been altered.  Please correct".format(self.training_file)
                return
        except:
            #functional
            self.error = "{} could not be opened or does not exist.  Please correct".format(self.training_file)
            return
        try:
            training_data = training_data.set_index("Compound Name")
            training_rt = training_data["RT"]
            descriptors = list(training_data.columns)
            descriptors.remove("RT")
            descriptors.remove("SMILES")
            training_data = training_data[descriptors]
        except:
            #functional
            self.error = "header of {} has been changed.  This is case sensitive.  Please correct.".format(self.training_file)
            return
        if len(descriptors) == 0:
            #functional
            self.error = "No descriptors present in {}.  Please correct".format(self.training_file)
            return
        #need to check is we have the the correct features and only the correct features 
        #we'll only bother if the features are chosen.  otherwise we can assume full mordred 
        #this is likely unnecessary, but covering our bases is a good idea.
        #if isinstance(self.ml_object.best_estimator_, sklearn.pipeline.Pipeline) and 'feature_selection' in self.ml_object.best_estimator_.named_steps.keys(): # need to check it's a pipeline (random forest isn't a pipeline if feature selection is used) and ensure feature_selection is present if it is a pipeline 
        if self.feature_selection_method == "Manual" or self.feature_selection_method == "Automatic":
            if type(chosen_features_file) == type(None): #this is not fool-proof since the user could be useing the same folder.  we will need another check
                #functional
                self.error = "There is no file containing chosen features when this is necessary for this model. Please correct"
                return
            self.needed_features = list(pd.read_csv(chosen_features_file)["Chosen Features"])
        
        if self.feature_selection_method == "Manual":
            #now we are sure that feature selection is necessary we can fiddle with files
            num_chosen_features = sum(self.ml_object["gridsearchcv_step"].best_estimator_.named_steps["feature_selection_step"].get_support()) #the number of features needed in the model
            if num_chosen_features != len(self.needed_features):
                #functional
                self.error = "Requried number of chosen features do not match the number of features in {}.  Please correct".format(chosen_features_file)
                return
            try:
                training_data[self.needed_features] #need to check that the features are present. don't actually reassign things.  or it will error out
            except KeyError:
                #functional
                self.error = "Features in {} were not present in {}.  Please correct".format(chosen_features_file, self.training_file)
                return
        elif self.feature_selection_method == "Automatic":
            try:
                #training_data = training_data[self.needed_features] #this was if rfecv is first.  now that it is part of the final pipeline we need to do a similar test as manual
                training_data[self.needed_features]
            except KeyError:
                self.error = "Features in {} were not the same as those in {}.  Some were missing or altered.  This is case sensitive.  Please correct.".format(chosen_features_file, self.training_file)
        #now that we have everything ready, we can do the actually testing
        try:
            current_score = self.ml_object.score(training_data ,training_rt)
        except ValueError:
            #functional
            self.error = "Training data is a different shape than during fitting.  This is likely due to a change in training file (usually adding or removing descriptor columns).  Please Correct."
            return
        if abs(current_score-self.training_score)/self.training_score > self.allowed_diffence_in_score:
            #functional
            self.error = "Training data and machine learning model scores do not agree. Files may have been changed or replaced.  Please correct."
            return 
        self.all_descriptors = list(training_data.columns)