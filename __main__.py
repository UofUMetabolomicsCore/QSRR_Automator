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

from copy import deepcopy
import multiprocessing as mp
import os, sys
from collections import Counter
import time  

import pandas as pd
from PyQt5 import QtCore, QtWidgets, uic

import generic_functions as gf
import create_machine_learning_model as cmlm
import assign_molecular_descriptors as amd
import collect_default_values as cdv
import settings_menu


location =  os.path.dirname(os.path.abspath(__file__))
#location = os.path.dirname(os.path.abspath(sys.executable))
SMILE_NAME = "SMILES"
RT_NAME = "RT"
NAME_NAME = "Compound Name"

Descriptor_Input_Headers = [NAME_NAME, RT_NAME, SMILE_NAME] #put this here so we can adjust the template creator and mordred query at once (will still demand specific columns, but allows adding extras easily)
Final_Input_Headers = [NAME_NAME, SMILE_NAME]

#need to collect more for the new model creation and the rt_predictions.  I believe we have everything from the main function calls but all the functions they call (make descriptors make the models, errors for the prediction) have not yet been analyzed.
NEW_MODEL_CREATION_POTENTIAL_FILENAMES_EXCEL_FILES = ["Training Processing Steps.xlsx", "Successful_Graph.pdf", "saved_model.joblib", "saved_model_settings.csv", "chosen_features.csv", "cross_validation_and_final_sores.csv", "Descriptors for training.xlsx"]
NEW_MODEL_CREATION_POTENTIAL_FILENAMES_CSV_FILES = ["Successful_Graph.pdf", "saved_model.joblib", "saved_model_settings.csv", "chosen_features.csv", "cross_validation_and_final_sores.csv", "Descriptors for training.csv", "Model building settings.csv", "Model building programmer settings.csv", "Model building numerical coercion.csv", "Model building bad samples removed.csv", "Model building Final Descriptors.csv", "Model building replicates removed.csv"]
RT_PREDICTIONS_POTENTIAL_FILENAMES_EXCEL_FILES =["Final RT Prediction.xlsx", "RT Prediction Steps.xlsx", "Descriptors for prediction.xlsx", "Full Error File.xlsx", "Relevant Error File.xlsx"]
RT_PREDICTIONS_POTENTIAL_FILENAMES_CSV_FILES =["Final RT Prediction.csv", "Descriptors for prediction.csv", "Full Error File.csv", "Relevant Error File.csv", "RT prediction settings.csv", "RT prediction programmer settings.csv", "RT prediction numerical coercion.csv"]

form_class = uic.loadUiType(os.path.join(location,"Generic_RT_Predictor_Main_Menu.ui"))[0]
class Main_Machine_Learning_Window(QtWidgets.QMainWindow, form_class):
    def __init__(self, parent = None):
        QtWidgets.QMainWindow.__init__(self,parent)
        self.setupUi(self)
        settings = cdv.get_default_values(location, True)
        if type(settings) == str:
            QtWidgets.QMessageBox.information(self, "Error", settings)
            #without the 2 closes running the code from a command line will stop allowing rerunning after the second error.  weird.  this fixes it, and since it doesn't have any negative effects, we'll just do both unless we figure out what is going on
            self.close()
            sys.exit(0)
        programmer_settings = cdv.get_default_values(location, False)
        if type(programmer_settings) == str:
            QtWidgets.QMessageBox.information(self, "Error", programmer_settings)
            #without the 2 closes running the code from a command line will stop allowing rerunning after the second error.  weird.  this fixes it, and since it doesn't have any negative effects, we'll just do both unless we figure out what is going on
            self.close()
            sys.exit(0)
        self.settings = settings
        self.default_settings = deepcopy(settings)
        self.programmer_settings = programmer_settings
        self.settings_series_to_write = gf.make_series_for_saving(self.settings)
        self.programmer_settings_series_to_write = gf.make_series_for_saving(self.programmer_settings)
        self.current_model = False
        self.current_model_descriptors = False
        
        self.program_location = location
        try:
            self.input_location = self.settings["Output Folder"]
            gf.make_folder(self.input_location)
            self.initial_output_loc = True
        except: #since input and output are the same, this will check permissions on both
            self.initial_output_loc = False
            #this only occurs if we have a permission error in making the new output folder.  in this case we can't trust that it exists and so will default to a location that must exist (the templates, output folder creator, new model and predict all have permission checks so having a location shouldn't cause an issue)
            # don't know why someone would put this in a admin locked folder but this should help
            self.input_location = self.program_location 
            self.settings["Output Folder"] = self.program_location 
        self.model_builder.clicked.connect(self.training_or_loading)
        self.RT_prediction_button.clicked.connect(self.rt_predictor)
        self.training_template_creator.clicked.connect(lambda: self.template_creator(True))
        self.prediction_template_creator.clicked.connect(lambda: self.template_creator(False))
        self.actionChoose_Output_Folder.triggered.connect(self.choose_output_folder)
        self.actionSettings.triggered.connect(self.settings_option)
        self.actionExit.triggered.connect(self.ExitButton)
        self.current_model_good = False
        self.loaded_model_feature_selection = False
        

    def check_for_overwriting(self, folder, files_to_check):
       what_will_be_overwritten = []
       for f in files_to_check:
           if os.path.isfile(os.path.join(folder, f)):
                what_will_be_overwritten.append(f)
       return what_will_be_overwritten
               
    def template_creator(self, need_rt):
        if self.settings["Excel Writing"]: priority = "XlSX (*.xlsx);; CSV (*.csv)"
        else: priority = "CSV (*.csv);; XlSX (*.xlsx)"
        if need_rt:template_panda = pd.DataFrame(columns =Descriptor_Input_Headers)
        else: template_panda = pd.DataFrame(columns = Final_Input_Headers)
        while(True):
            #will need to deal saving.  we'll see if this does it right
            QtWidgets.QMessageBox.information(self, "Info", "Please select the filename and location to save your template.")
            save_file, file_type = QtWidgets.QFileDialog.getSaveFileName(self, "Provide Save File", self.input_location, priority)#will ask if you wish to overwrite and add the .cef tag if necessary
            if save_file =="":
                return #will break out of the while without needing to break
            try:
                if file_type == "CSV (*.csv)": template_panda.to_csv(save_file, index = False)
                elif file_type =="XlSX (*.xlsx)": template_panda.to_excel(save_file, index = False)
                self.input_location = os.path.dirname(save_file)
                QtWidgets.QMessageBox.information(self, "Info", "Your template was successfully created")
                break
            except:
                #note that windows seems to do this on it's own, but having this here is harmless in case windows fails and I don't like trusting another software more than I don't like have an extra while and try in here.
                QtWidgets.QMessageBox.information(self, "Error", "File {} cannot be written likely due to permission issue in folder {}. Please try again in a different location".format(save_file, os.path.dirname(save_file)))
        
        
    def descriptor_assignment(self, need_rt, input_file, excel_writer_object = None):
        to_mordred, input_dataframe = amd.initial_preparation_machine_learning_input(input_file, self.settings, self.programmer_settings, need_rt, self.current_model_descriptors, self.loaded_model_feature_selection)
        if type(to_mordred) == str: return to_mordred, "" #will be an error message if a string
        #quick test to see if user wants to proceed if they have one compound with multipe RTs
        if need_rt:
            instances_of_smiles = Counter(input_dataframe[SMILE_NAME])
            potential_issues = [k for k in instances_of_smiles.keys() if instances_of_smiles[k] >1]
            #smiles are the same so this is not going to be complex. if there are more rts than names, then there is a duplicate name without a duplicate rt
            #won't have the specific target but should work for a basic "Should we continue" warning
            for p in potential_issues:
                small_df = input_dataframe[input_dataframe[SMILE_NAME] == p]
                temp_names = list(small_df[NAME_NAME])
                temp_rts=list(small_df[RT_NAME])
                if len(set(temp_names)) < len(set(temp_rts)):
                    reply = QtWidgets.QMessageBox.question(self, "Warning", "There are multiple instances of a compound with different Retention Times.  This can cause poor machine learning fits or have no real effect.  Would you like to continue?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
                    if reply == QtWidgets.QMessageBox.No:
                        return "", ""
                    else:
                        break
        #in this case we have extra columns we are going to assume are descriptors.  confirm with the user.  if so great, don't do mordred (though still do error checks.  otherwise purge these columns 
        if not to_mordred:
            custom_msg_box = QtWidgets.QMessageBox()
            custom_msg_box.setText("You already have descriptors in this input file.  Do you wish to use them or recalculate?")
            cancel_button = custom_msg_box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole) #needed because otherwise the red x will actually do one of the buttons, which the user likely doesn't want.
            keep_button = custom_msg_box.addButton("Keep Existing", QtWidgets.QMessageBox.YesRole)
            recalculate_button = custom_msg_box.addButton("Recalculate", QtWidgets.QMessageBox.NoRole)
            custom_msg_box.exec_()
            if custom_msg_box.clickedButton() == cancel_button:
                return None, None
            elif custom_msg_box.clickedButton() == keep_button:
                to_mordred = False
            elif custom_msg_box.clickedButton() == recalculate_button:
                to_mordred = True
                #if we're recalculating purge the other columns
                if need_rt:
                    input_dataframe = input_dataframe[Descriptor_Input_Headers]
                else:
                    input_dataframe = input_dataframe[Final_Input_Headers]
        if to_mordred: 
            numerical_df = amd.actually_add_descriptors(input_dataframe)
            if type(numerical_df) == str: return numerical_df, ""
        #if we are assuming things are correct we need to remove RT and names and make the smiles the index and 
        else:
            numerical_df = input_dataframe.set_index(SMILE_NAME)
            numerical_df = numerical_df.drop(NAME_NAME, axis = 1)
            
            if need_rt: numerical_df = numerical_df.drop(RT_NAME, axis = 1)
            #need to drop duplicates of smiles with same descriptors (can cause issues with filters)
            numerical_df = numerical_df.drop_duplicates()
            #need to ensure that all duplicate smiles are removed (if it didn't drop, smiles are same but descriptors are not.  RT is dropped at this point)
            instances_of_smiles = Counter(numerical_df.index)
            problematic_smiles = [k for k in instances_of_smiles.keys() if instances_of_smiles[k] >1]
            if problematic_smiles:
                return "The following SMILES have duplicate SMILES with different descriptors (retention time not included): {}. Please correct to continue".format(problematic_smiles), None
            if need_rt: input_dataframe = input_dataframe[Descriptor_Input_Headers]
            else: input_dataframe = input_dataframe[Final_Input_Headers]
        numerical_df = numerical_df.apply(pd.to_numeric, args = ['coerce'])
        #now we need to actually write out the data.  need to make the excel object (write out the end separately as well so they can use that as a read in)
        if self.settings["Excel Writing"]:
            input_dataframe.to_excel(excel_writer_object, "Initial Input", index = False)
            self.settings_series_to_write.to_excel(excel_writer_object, "User Settings")
            self.programmer_settings_series_to_write.to_excel(excel_writer_object, "Programmer Settings")
            numerical_df.to_excel(excel_writer_object, "Added Descriptors")
        else:
            things_to_write = [self.settings_series_to_write, self.programmer_settings_series_to_write, numerical_df]
            if need_rt: file_names = ["Model building settings.csv", "Model building programmer settings.csv", "Model building numerical coercion.csv"]
            else: file_names = ["RT prediction settings.csv", "RT prediction programmer settings.csv", "RT prediction numerical coercion.csv"]
            for i in range(len(things_to_write)):
                write_attempt = gf.write_single_file(os.path.join(self.settings["Output Folder"], file_names[i]), things_to_write[i], True, False)
                if write_attempt != "Success": return write_attempt, ""
        if need_rt:
            if self.settings["Excel Writing"]:
                trimmed_dataframe = amd.filter_descriptors(numerical_df, self.settings, self.programmer_settings, self.program_location, excel_writer_object)
            else:
                trimmed_dataframe = amd.filter_descriptors(numerical_df, self.settings, self.programmer_settings, self.program_location)     
            if type(trimmed_dataframe) == str:
                return trimmed_dataframe, ""
            trimmed_dataframe = pd.merge(input_dataframe, trimmed_dataframe, right_index = True, left_on = SMILE_NAME, how = 'inner')
            if self.settings["Excel Writing"]:
                final_filename = os.path.join(self.settings["Output Folder"], "Descriptors for training.xlsx")
                trimmed_dataframe.to_excel(excel_writer_object, "Final Descriptor Output", index = False)
                write_error = gf.write_single_file(final_filename, trimmed_dataframe, False, True)
            else:
                final_filename =os.path.join(self.settings["Output Folder"], "Descriptors for training.csv")
                write_error = gf.write_single_file(final_filename, trimmed_dataframe, False, False)
            if write_error != "Success": return write_error, ""
        else:
            try:
                descriptor_trimmed_dataframe = numerical_df[self.current_model_descriptors]
            except KeyError:
                return "Required descriptors for your model not in calculated descriptors", ""
            trimmed_dataframe = pd.merge(input_dataframe, descriptor_trimmed_dataframe, right_index = True, left_on = SMILE_NAME, how = 'inner')
            if self.settings["Excel Writing"]:
                final_filename = os.path.join(self.settings["Output Folder"], "Descriptors for prediction.xlsx")
                trimmed_dataframe.to_excel(excel_writer_object, "Final Descriptor Output", index = False)
                write_error = gf.write_single_file(final_filename, trimmed_dataframe, False, True)
            else:
                final_filename = os.path.join(self.settings["Output Folder"], "Descriptors for prediction.csv")
                write_error = gf.write_single_file(final_filename, trimmed_dataframe, False, False)
                if write_error != "Success": return write_error, ""
            if descriptor_trimmed_dataframe.isnull().values.any():
                #write out all the descriptors in all cases in case the user wishes to know
                true_false_df = pd.merge(input_dataframe, descriptor_trimmed_dataframe.isnull(), right_index = True, left_on = SMILE_NAME, how = 'inner')
                if self.settings["Excel Writing"]:
                    problem_file = os.path.join(self.settings["Output Folder"], "Full Error File.xlsx")
                    gf.write_single_file(problem_file,true_false_df, False, True)
                else:
                    problem_file = os.path.join(self.settings["Output Folder"], "Full Error File.csv")
                    gf.write_single_file(problem_file,true_false_df, False, False)
                #$we're going to test if the missing compounds are actually necessary.  if not we'll fill them with a dummy and see what happens
                #$ to undo this get rid of the if and all under it and put the return in the else back a tab
                # first we need the actual descriptors we need
                if self.loaded_model_feature_selection == "Automatic" or self.loaded_model_feature_selection == "Manual": #with None everything that remains should be good since there is no feature_selection_step
                    #need the true/false mask in order to find the chosen features
                    try: 
                        chosen_value_mask = list(self.current_model.named_steps["feature_selection_step"].support_)
                    except KeyError: 
                       chosen_value_mask = list(self.current_model.named_steps["gridsearchcv_step"].best_estimator_.named_steps["feature_selection_step"].get_support())
                    #turn the mask into a  list of columns (otherwise it will be interpreted as an index in the next step instead of columns)
                    chosen_compounds = descriptor_trimmed_dataframe.columns[chosen_value_mask]
                    #data frame of only the needed compounds for the model
                    only_chosen_values = descriptor_trimmed_dataframe[chosen_compounds]
                    if only_chosen_values.isnull().values.any():#if the needed compounds have nans we can't predict (we could drop, but the user is better able to determine what needs to be done here)
                        #if there is a problem we need to help the user find the problem (otherwise we may end up with many failed descriptors that aren't needed by the model)
                        true_false_df = pd.merge(input_dataframe, only_chosen_values.isnull(), right_index = True, left_on = SMILE_NAME, how = 'inner')
                        if self.settings["Excel Writing"]:
                            problem_file = os.path.join(self.settings["Output Folder"], "Relevant Error File.xlsx")
                            gf.write_single_file(problem_file,true_false_df, False, True)
                        else:
                            problem_file = os.path.join(self.settings["Output Folder"], "Relevant Error File.csv")
                        
                        return "some compounds were missing values for needed descriptors.  Please look at {} for the problematic features (all should be False.  True indicates a problem). Please remove the problematic compounds or retrain the model and try again.".format(problem_file), ""
                    else: #we will give a warning and then prevent further erroring out
                        reply = QtWidgets.QMessageBox.information(self, "Warning", "There are compounds which are lacking descriptors found in your training data.  These features are not used in your model so prediction will proceed, but this may indicate problematic differences between compounds.  Can observe which descriptors have an issue in {} (all should be False.  True indicates a problem)".format(problem_file))
                        trimmed_dataframe = trimmed_dataframe.fillna(0) #allows us to proceed without issues
                else:
                    return "some compounds were missing values for needed descriptors.  Please look at {} for the problematic features (all should be False.  True indicates a problem). Please remove the problematic compounds or retrain the model and try again.".format(problem_file), ""        
        return trimmed_dataframe, final_filename
                                                
    def training_or_loading(self):
        #question msg box won't cut it.  so we'll need to make a new one.
        #basing this off of https://stackoverflow.com/questions/49155926/how-to-customise-a-pyqt-message-box accessed 12/14/18 answer 1
        #rebuilding each time should take minimal effort for the processor.  if it takes too long can build in a seperate function called once by the __init__ then just call the exec_ here
        custom_msg_box = QtWidgets.QMessageBox()
        custom_msg_box.setText("Do you want to train a New Method or Load a previous Method?")
        new_button = custom_msg_box.addButton("New Model", QtWidgets.QMessageBox.YesRole)
        load_button = custom_msg_box.addButton("Load Model", QtWidgets.QMessageBox.NoRole)
        
        custom_msg_box.exec_()
        
        #don't use else.  if nothing is clicked (they hit the x in the top right) they should not be channeled into any function
        if custom_msg_box.clickedButton() == new_button:
            if not self.initial_output_loc:
                QtWidgets.QMessageBox.information(self, "Error", "QSRR Automator lacks permission to write in current output folder.  Please use the \"Choose Output Folder\" folder in the pull down menu in the top left corner to select a new folder and try again")
                return
            self.train_the_model()
        elif custom_msg_box.clickedButton() == load_button:
            self.load_a_model()
                
    def train_the_model(self):
        #first thing we need to do is warn the user if we are going to overwrite a file
        if self.settings["Excel Writing"]:
            overwrite_warning_list = self.check_for_overwriting(self.settings["Output Folder"], NEW_MODEL_CREATION_POTENTIAL_FILENAMES_EXCEL_FILES)
        else:
            overwrite_warning_list = self.check_for_overwriting(self.settings["Output Folder"], NEW_MODEL_CREATION_POTENTIAL_FILENAMES_CSV_FILES)
        if overwrite_warning_list != []:
            reply = QtWidgets.QMessageBox.question(self, "Overwrite Warning", "The following files are located in folder {} may be overwritten if you proceed in this folder:\n{}\nDo you wish to continue?".format(self.settings["Output Folder"], "\n".join(overwrite_warning_list)), QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.No:
                return 
            
        start = time.time()
        input_file, filter_ = QtWidgets.QFileDialog.getOpenFileName(self, "Select filled in \"Training Template\" file", self.settings["Output Folder"], "(*.csv *.xlsx)")
        if not input_file:
            return 
        self.input_location = os.path.dirname(input_file)
        if self.settings["Excel Writing"]:
            excel_writer = pd.ExcelWriter(os.path.join(self.settings["Output Folder"],"Training Processing Steps.xlsx"))
            training_data, final_filename = self.descriptor_assignment(True, input_file, excel_writer)
        else:
            training_data, final_filename = self.descriptor_assignment(True, input_file)
        if type(training_data) == type(None): #straight comparison to None errors out if training_data is a dataframe so this will cover all situations
            if self.settings["Excel Writing"]: gf.deal_with_writer_object(excel_writer, True)
            return
        elif type(training_data) == str:
            if training_data != "":
                QtWidgets.QMessageBox.information(self, "Error", training_data)
            self.current_model_good = False
            self.loaded_model_feature_selection = False
            self.current_model = False
            self.current_model_descriptors = False
            if self.settings["Excel Writing"]: gf.deal_with_writer_object(excel_writer, True)
            return
        user_save, model_object, series_to_save, score_series = cmlm.model_training_manager(training_data, final_filename, self.settings, self.programmer_settings) # returns a model object or an error message
        if not user_save: #only triggered by a specific user choice with a warning
            if self.settings["Excel Writing"]: gf.deal_with_writer_object(excel_writer)
            return
        """#this is unnecessary since there are no error messages (and the only failure condition is dealt with using the user_save above.
        #if this changes for whatever reason
        if type(model_object) == str: #if it's a string, it's an error message not a gridsearchcv object
            QtWidgets.QMessageBox.information(self, "Error", model_object)
            self.current_model_good = False
            self.current_model = False
            self.current_model_descriptors = False
        else:"""
        self.current_model = model_object.main_pipeline
        self.current_model_descriptors = model_object.all_required_features
        if self.settings["Excel Writing"]:
            failures = cmlm.save_data(model_object, self.settings, series_to_save, score_series, excel_writer)
        else:
            failures = cmlm.save_data(model_object, self.settings, series_to_save, score_series)
        if failures != "Success":
            QtWidgets.QMessageBox.information(self, "Error", failures)
            if self.settings["Excel Writing"]: gf.deal_with_writer_object(excel_writer, True)
            return
        if self.settings["Excel Writing"]: 
            stuff = gf.deal_with_writer_object(excel_writer)
            if stuff == "Permission Error":
                QtWidgets.QMessageBox.information(self, "Error", "{} is open in another program.  Please close and try again.".format(os.path.join(self.settings["Output Folder"],"Training Processing Steps.xlsx"))) #$ may need to add a proceed option here
                return
        QtWidgets.QMessageBox.information(self, "Success", "Model has been trained and saved.  Time taken: {} min".format((time.time()-start)/60)) #$ may need to add a proceed option here
        self.loaded_model_feature_selection = self.settings["Model To Use"]
        self.current_model_good = True
    
    def load_a_model(self):
        gf.make_folder(self.settings["Output Folder"])
        QtWidgets.QMessageBox.information(self, "Info", "Please choose the folder with your model in it.  This will be the export folder of the run you trained the model in")
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a Folder", self.settings["Output Folder"], QtWidgets.QFileDialog.ShowDirsOnly)
        if not folder: return
        error, descriptors, feature_selection_method_used = cmlm.load_data(folder, self.programmer_settings)
        if type(error) == str:
             QtWidgets.QMessageBox.information(self, "Error", error)
             self.current_model_good = False
             self.loaded_model_feature_selection = False
             self.current_model = False
             self.current_model_descriptors = False
        else:
             self.current_model = error.ml_object
             self.current_model_descriptors = descriptors
             self.current_model_good = True
             self.loaded_model_feature_selection = feature_selection_method_used
             QtWidgets.QMessageBox.information(self, "Success", "Your model was loaded successfully")
        
    
    
    def rt_predictor(self):
        if not self.current_model_good:
            QtWidgets.QMessageBox.information(self, "Error", "You have not built or loaded a model, or your latest attempt to do so was unsuccessful.  Please successfully build or load a model before proceeding.")
            return
        #only if the output folder is in an admin locked location. if the output location is changed this check is not tripped.  likely to be completely unnecessary but one if won't slow us too much
        if not self.initial_output_loc:
            QtWidgets.QMessageBox.information(self, "Error", "QSRR Automator lacks permission to write in current output folder.  Please use the \"Choose Output Folder\" folder in the pull down menu in the top left corner to select a new folder and try again")
            return
            
        #need to warn the user if we are going to overwrite a file
        if self.settings["Excel Writing"]:
            overwrite_warning_list = self.check_for_overwriting(self.settings["Output Folder"], RT_PREDICTIONS_POTENTIAL_FILENAMES_EXCEL_FILES)
        else:
            overwrite_warning_list = self.check_for_overwriting(self.settings["Output Folder"], RT_PREDICTIONS_POTENTIAL_FILENAMES_CSV_FILES)
        if overwrite_warning_list != []:
            reply = QtWidgets.QMessageBox.question(self, "Overwrite Warning", "The following files are located in folder {} may be overwritten if you proceed in this folder:\n{}\nDo you wish to continue?".format(self.settings["Output Folder"], "\n".join(overwrite_warning_list)), QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.No:
                return
                    
        #need to get the data 
        gf.make_folder(self.settings["Output Folder"])
        QtWidgets.QMessageBox.information(self, "Info", "Please choose file containing compounds for RT prediction.  this should be based on the output of the \"Prediction Template\" button")
        input_file, filter_ = QtWidgets.QFileDialog.getOpenFileName(self, "Select RT prediction input file", self.settings["Output Folder"], "(*.csv *.xlsx)")
        if not input_file:
            return 
        self.input_location = os.path.dirname(input_file)
        if self.settings["Excel Writing"]:
            excel_writer = pd.ExcelWriter(os.path.join(self.settings["Output Folder"],"RT Prediction Steps.xlsx"))
            test_data, final_filename = self.descriptor_assignment(False,input_file, excel_writer)
        else:
            test_data, final_filename = self.descriptor_assignment(False, input_file)
        if type(test_data) == type(None): #a dataframe will error out if compared to None and this may not be a dataframe so can't use pandas methods. this works fine
            if self.settings["Excel Writing"]: gf.deal_with_writer_object(excel_writer, True)
            return
        elif type(test_data) == str:
            if test_data != "":
                QtWidgets.QMessageBox.information(self, "Error", test_data)
            if self.settings["Excel Writing"]: gf.deal_with_writer_object(excel_writer)
            return
        #need to adjust with features if we have such
        predicted_rt = cmlm.analyze_user_data(test_data, self.current_model, self.settings)
        predicted_data = test_data[Final_Input_Headers].copy() #the .copy() notation tells pandas this is a copy, preventing the setting with copy warning on the next line without deactivating the warning behavior
        predicted_data[RT_NAME] =predicted_rt
        if self.settings["Excel Writing"]:
            predicted_data.to_excel(excel_writer, "RT prediction", index = False)
            excel_error = gf.deal_with_writer_object(excel_writer)
            if  excel_error != None:
                QtWidgets.QMessageBox.information(self, "Error", "{} is open in another program.  Please close and try again or change settings.".format(os.path.join(self.settings["Output Folder"],"RT Prediction Steps.xlsx")))
                return
            error = gf.write_single_file(os.path.join(self.settings["Output Folder"], "Final RT Prediction.xlsx"), predicted_data, False, True) 
        else:
            error = gf.write_single_file(os.path.join(self.settings["Output Folder"], "Final RT Prediction.csv"), predicted_data, False, False)
        if error == "Success":
            QtWidgets.QMessageBox.information(self, "Success", "Your retention times were predicted.")
        else:
            QtWidgets.QMessageBox.information(self, "Error", error)
            
    #functional     
    def choose_output_folder(self):
        gf.make_folder(self.settings["Output Folder"])
        empty_df_for_testing = pd.DataFrame()
        while(True):
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a Folder", self.settings["Output Folder"], QtWidgets.QFileDialog.ShowDirsOnly)
            if folder:
                try:
                    empty_df_for_testing.to_csv(os.path.join(folder, "generic_output_testing_file.csv"))
                    os.remove(os.path.join(folder, "generic_output_testing_file.csv"))
                    self.settings["Output Folder"] = folder
                    self.input_location = folder
                    self.initial_output_loc = True
                    break
                except:
                    QtWidgets.QMessageBox.information(self, "Error", "Folder {} cannot be written to likely due to permission issue. Please move output folder to a location that can be written to proceed".format(folder))
            else:
                break
                
    
    def settings_option(self):
        self.set_menu = settings_menu.Settings_Menu(self, self.settings, self.default_settings)
        self.set_menu.show()
        
        
    def ExitButton(self):
        # from http://stackoverflow.com/questions/1414781/prompt-on-exit-in-pyqt-application
        reply = QtWidgets.QMessageBox.question(self, "Quit Option", "Are you sure you wish to exit?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.close()
            
if __name__ == "__main__":
    mp.freeze_support()# this is critical since mordred and random forest use multiprocessing. 
    app = QtWidgets.QApplication(sys.argv)
    interaction_gui = Main_Machine_Learning_Window(None)
    interaction_gui.show()
    app.exec_()