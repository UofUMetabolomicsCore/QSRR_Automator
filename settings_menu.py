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

#runs the setting menu to adjust user settings.
#if we need to put in filters to check the defaults to ensure they are valid can put here instead of settings_class_module to keep the limits togther (as separate functions not in the main class)

#make sure most ints are above 0 and turn percents to decimals
#turn most text to boolean
#ensure that if feature selection is quite large we actually have that many features (do in the appropriate area) 

import os, sys
import multiprocessing as mp

from PyQt5 import QtCore, QtWidgets, uic

location =  os.path.dirname(os.path.abspath(__file__))
#location = os.path.dirname(os.path.abspath(sys.executable))

form_class = uic.loadUiType(os.path.join(location,"Settings_Menu.ui"))[0]
class Settings_Menu (QtWidgets.QMainWindow, form_class):
    def __init__(self, parent = None, current_settings = None, default_settings = None):
        super(Settings_Menu, self).__init__(parent)
        self.current_settings = current_settings # due to how python works, altering this will alter the settings in the main so we don't have to fiddle with 
        self.setWindowTitle("Settings Menu")
        self.setupUi(self)
        self.processors_to_use.setMaximum(mp.cpu_count()) # set the maximum number of processors so we don't assign too many processors (doing so will slow down the calculations)
        #need to set the current values as the currently default values
        self.load_data(self.current_settings)
        self.SaveButton.clicked.connect(self.save_data)
        self.RestoreLastSaveButton.clicked.connect(lambda: self.load_data(self.current_settings))
        self.ResetDefaultButton.clicked.connect(lambda: self.load_data(default_settings))
        self.ExitButton.clicked.connect(self.close) #may wish to check for changes and then give an option to close or not (if do so link that to the close so it catches  the red x as well
    
    
    def boolean_parser(self, gui_object, true_value):
        if str(gui_object.currentText()) == true_value:
            return True
        else:
            return False
            
    
    def min_max_checker(self, minimum, maximum, number_of_steps, name, int_check = False):
        if minimum > maximum:
            return "Minimum {} is greater than Maximum {}.  Please correct in order to save.".format(name, name), "", "", ""
        #would like a special message for the min and max being equal since it is needed for both (C and gamma can have a near infinite number of steps so long as the numbers are not the same
        if minimum == maximum and number_of_steps != 1:
            return "Minimum {} and Maximum {} are the same, but number of steps is not 1.  Cannot have multiple steps between the same values.  Please correct and try again".format(name, name), "", "", ""
        if minimum != maximum and number_of_steps == 1:
            return "Minimum {} and Maximum {} are different but number of steps is 1.  Please make Minimum and Maximum the same or increase number of steps.".format(name, name), "", "", ""
        if int_check:
            if maximum-minimum < number_of_steps -1: #first, last and any integers between.  3-1 =2, so 3-1 steps is valid.  4 steps requires decimals.
                return "Minimum {} and Maximum {} are close enough that it would require decimals to have provided number of steps.  Please make Minimum and Maximum the same or increase number of steps.".format(name, name), "", "", ""
        return "", minimum, maximum, number_of_steps 
              
    
    def bool_string_adjuster(self, combo_box, yes_value, no_value, value):
        if value:
            needed_value = yes_value
        else:
            needed_value = no_value
        index = combo_box.findText(needed_value)
        combo_box.setCurrentIndex(index)
    
    #since some values have requirements (min should be less than max, there should be num of steps ints between min and max, etc.)  we need to use some if checks
    # don't need to check for ints and so on since these should be dealt with by the nature of the gui.  therefore we can 
    def save_data(self, checking = False):
        #because we can't save the data until we are sure all is good, we'll save in a dict first.   return if any checks fail.  if succeed, updating the pandas dataframe will only take a quick loop
        temp_dict = {}
        #we'll group the non-numericals first 
        #booleans
        temp_dict["Excel Writing"] = self.boolean_parser(self.Output_type, "Excel")
        #string
        temp_dict["Feature Selection Method"] = str(self.feature_selection_option.currentText())
        temp_dict["Model To Use"] = str(self.machine_learning_model.currentText()) 
        #numericals from General settings, Random Forest Settings and Descriptor settings we can trust the filters that are part of the spin boxes
        temp_dict["Number of Cross Validations"] = int(self.num_cross_validations.value())
        temp_dict["Automatic Feature Selection CV"] = int(self.auto_cross_validations.value())
        temp_dict["Void volume cutoff"] = float(self.void_volume_box.value())
        #processors need a warning if we try to use all processors (though if only 1 processor no point in the warning since they have no choice)
        if int(self.processors_to_use.value()) == mp.cpu_count() and mp.cpu_count != 1 and not checking:
            reply = QtWidgets.QMessageBox.question(self, "All Processors", "You have selected to use all processors.  This will speed up caclulations, but make it very difficult to use your computer for anything else.  Is this correct?", QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.No:
                return
        temp_dict["Processors_to_Use"] = int(self.processors_to_use.value())
        
        
        #the random forest  need a few adjustments to undo decimals
        temp_dict["Allowed same value per descriptor"] = float(self.allowed_descriptor_same_value.value())/100.0
        temp_dict["NaNs allowed per sample"] = float(self.allowed_missing_descriptors_per_sample.value())/100.0
        temp_dict["Correlation Coefficient (Pearson)"] = float(self.maximum_correlation_allowed.value())#pearson r is already a decimal
        #SVR settings need a few extra checks, thankfully they are the same for two bathces. if these error out we can raise and error and return
        error, temp_dict["C min"], temp_dict["C max"], temp_dict["C number of steps"] = self.min_max_checker(float(self.min_C_value.value()), float(self.max_C_value.value()), int(self.number_C_steps.value()), "C")
        if error != "":
            if not checking: QtWidgets.QMessageBox.information(self, "Error", error)
            return
        error, temp_dict["gamma min"], temp_dict["gamma max"], temp_dict["gamma number of steps"] = self.min_max_checker(float(self.min_gamma_value.value()), float(self.max_gamma_value.value()), int(self.number_gamma_steps.value()), "gamma")
        if error != "":
            if not checking: QtWidgets.QMessageBox.information(self, "Error", error)
            return
        temp_dict["Min Number of Features"] = int(self.min_number_of_features.value())
        temp_dict["Target Number Features for Manual"] = int(self.number_of_target_features_manual.value())
        if error != "":
            if not checking: QtWidgets.QMessageBox.information(self, "Error", error)
            return
        if checking:
            return temp_dict
        #now that everything is good, we can actually save (if this fails should error out.  no way this can be the user's fault unless they were mucking with the default files or code or something
        for key in temp_dict.keys():
            self.current_settings[key] = temp_dict[key]
        QtWidgets.QMessageBox.information(self, "Success", "Settings successfuly saved")
    #we'll load the values with this function     
    def load_data(self, settings_series):
        #boolean adjustment
        self.bool_string_adjuster(self.Output_type, "Excel", "CSV", settings_series["Excel Writing"])
        #string adjustment
        index = self.machine_learning_model.findText(settings_series["Model To Use"])
        self.machine_learning_model.setCurrentIndex(index)
        index2 = self.feature_selection_option.findText(settings_series["Feature Selection Method"])
        self.feature_selection_option.setCurrentIndex(index2)
        #numerical values without fiddling with percentatges
        numerical_objects = {"Void volume cutoff": self.void_volume_box,"Number of Cross Validations":self.num_cross_validations, "Min Number of Features": self.min_number_of_features, "Target Number Features for Manual": self.number_of_target_features_manual, "Automatic Feature Selection CV": self.auto_cross_validations, "Correlation Coefficient (Pearson)":self.maximum_correlation_allowed, "C min":self.min_C_value, "C max": self.max_C_value, "C number of steps":self.number_C_steps, "gamma min": self.min_gamma_value,"gamma max":self.max_gamma_value, "gamma number of steps":self.number_gamma_steps}
        for n in numerical_objects.keys():
             numerical_objects[n].setValue(settings_series[n])
        #processors need a value check (for first loading)
        if settings_series["Processors_to_Use"] > mp.cpu_count():
            self.processors_to_use.setValue(mp.cpu_count())
        else:
            self.processors_to_use.setValue(settings_series["Processors_to_Use"])
        percent_adjustment_needed = {"Allowed same value per descriptor":self.allowed_descriptor_same_value, "NaNs allowed per sample": self.allowed_missing_descriptors_per_sample}
        for p in percent_adjustment_needed.keys():
            percent_adjustment_needed[p].setValue(settings_series[p]*100)
        
    def check_if_save_is_good(self):
        values_in_menu = self.save_data(True)
        if values_in_menu == None:
            return False, False
        for key in values_in_menu.keys():
            if self.current_settings[key] != values_in_menu[key]:
                return False, True
        return True,True
    
    def closeEvent(self, event):
        already_saved, no_errors_in_save_check = self.check_if_save_is_good()
        if already_saved:
            event.accept()
        elif no_errors_in_save_check:
            reply = QtWidgets.QMessageBox.question(self, "Save", "Changes have been made since your last save.  Would you like to save before you exit?", QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No|QtWidgets.QMessageBox.Cancel)
            if reply == QtWidgets.QMessageBox.No:
                event.accept()        
            elif reply == QtWidgets.QMessageBox.Yes:
                self.save_data()
                event.accept()
            else:
                event.ignore()
        else:
            reply = QtWidgets.QMessageBox.question(self, "Warning", "Your settings are changed but cannot be saved due to unallowed values of some type.  (\"The Save Settings\" button will provide details on these unallowed values.)  Would you like to exit without saving?" , QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                event.accept()        
            else:
                event.ignore()

#this is for running code independently (for testing purposes)
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = Settings_Menu()
    myapp.show()
    sys.exit(app.exec_())