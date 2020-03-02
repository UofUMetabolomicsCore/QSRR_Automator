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
since the settings are based on .csvs to make it easy for the user or future programmers to modify we need to prevent issues.
the main issue is avoiding incorrect types, invalid values and so on.  This prevents the need for tons of ifs or try excepts whenever we use the a setting or programmer setting
this is mostly a port of previous code created for Reportorator.  a bit more complicated because of the need for two option sets but is otherwise similar
"""
import csv
import os
import multiprocessing

USER_SETTINGS_FILE = "User_Settings_Defaults.csv"
PROGRAMER_SETTINGS_FILE = "Programmer_Settings_Defaults.csv"

class String_Values(object):
    #name is the name of the variable, lowere needed is a boolean to determine is we should force it to lower (case insensitive), allowed_values is a list of allowed values for this string, and boolean is a boolean to say if this should be turned into a boolean (True) or left as a string (False)
    def __init__(self, name, lower_needed, allowed_values, boolean):
        self.name = name
        self.lower_needed = lower_needed
        self.allowed_values = allowed_values
        self.boolean = boolean
        self.error = "" #if this get's filled we have an issue with this string
    def check_value (self, value):
        value = str(value)
        if self.lower_needed:
            value = value.lower()
        if value not in self.allowed_values:
            #!functional
            self.error = "\"{}\" not a valid default value for variable \"{}\".  Please change to a variable from the allowed list: {}".format(value, self.name, self.allowed_values)
        elif self.boolean:
            if value == "true":
                self.value = True
            elif value == "false":
                self.value = False
        else:
            self.value = value

#this is a super class for int and float sub classes.  the sub classes actually try to force the values  into the proper variable types with appropriate error messages for failure
#this governs things like too low or too high
class Numerical_Values(object):
    def __init__(self, name, minimum, maximum):
        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        self.error = ""
    def check_value(self,value):
        if value < self.minimum:
            #!functional
            self.error = "{}, the default value provided for \"{}\" is less than the allowed minimum of {}".format(value, self.name, self.minimum)
        elif value > self.maximum:
            #!functional
            self.error = "{}, the default value provided for \"{}\" is greater than the allowed maximum of {}".format(value, self.name, self.maximum)
        else:
            self.value = value

class Float_Values (Numerical_Values):
    def check_value(self, value):
        try:
            value = float(value)
        except ValueError:
            #!functional
            self.error = "{}, the default value provided for \"{}\" could not be converted to a floating point value.  Please correct.".format(value, self.name)
            return #no point in dealing with anything else if the error is already established
        super().check_value(value)
        
class Int_Values (Numerical_Values):
    def check_value(self, value):
        try:
            value = int(value)
        except ValueError:
            #!functional
            self.error = "{} the default value provided for \"{}\" could not be converted to an integer.  Please correct.".format(value, self.name)
            return #no point in dealing with anything else if the error is already established
        super().check_value(value)
        
#for most of the elements in these dictionaries the minimum and maximum values are not reasonable.   the goal is to prevent entirely stupid values (negative values, percents greater than 100% on a filter)
#we also want to avoid values outside of those allowed by the settings gui so that it does not error out

user_settings_and_values = {
    "Void volume cutoff": Float_Values("Void volume cutoff", 0.00, 9999.99),
    "Allowed same value per descriptor": Float_Values("Allowed same value per descriptor", .1, .99),
    "NaNs allowed per sample": Float_Values("NaNs allowed per sample", 0.0, .90),
    "Correlation Coefficient (Pearson)": Float_Values("Correlation Coefficient (Pearson)", .05, 1.00),
    "Excel Writing": String_Values("Excel Writing", True, ["true", "false"], True), 
    "Model To Use": String_Values("Model To Use", False, ["Choose Best", "Random Forest", "SVR", "MLR"], False),
    "Number of Cross Validations": Int_Values("Number of Cross Validations", 5, 10),
    "Feature Selection Method": String_Values("Feature Selection Method", False, ["Automatic", "None", "Manual"], False),
    "Automatic Feature Selection CV": Int_Values("Automatic Feature Selection CV", 5, 10),
    "Min Number of Features": Int_Values("Min Number of Features", 5, 9999),
    "Target Number Features for Manual": Int_Values("Target Number Features for Manual", 5, 9999),
    "C min": Float_Values("C min", 0.000100, 99999.0),
    "C max": Float_Values("C max", 0.000100, 99999.0),
    "C number of steps": Int_Values("C number of steps", 1, 99),
    "gamma min": Float_Values("gamma min", 0.000100, 99999.0),
    "gamma max": Float_Values("gamma max", 0.000100, 99999.0),
    "gamma number of steps": Int_Values("gamma number of steps", 1, 99)
}
programmer_settings_and_values = {
    "Trees in Random Forest": Int_Values("Trees in Random Forest", 5, 100000),
    "Allowed Model Loading Error": Float_Values("Allowed Model Loading Error", .001, .2),
    "Minimum Machine Learning Descriptors": Int_Values("Minimum Machine Learning Descriptors", 3, 1000),
    "Auto Feauture Selection Step Size": Int_Values("Auto Feauture Selection Step Size", 1, 20),
    "Minimum Number of Samples Allowed": Int_Values("Minimum Number of Samples Allowed", 50, 1000)
}


#there are a few min max pairs we need to ensure that the minimum is <= the max.  this is based of of the min_max_checker function in the settings_menu.py
#there are in order of minimum, maximum, step size, and if steps between them must be integers
user_settings_min_max = [["C min", "C max", "C number of steps", False], ["gamma min", "gamma max", "gamma number of steps", False]]

def deal_with_duplicates(dictionary_to_check, list_of_duplicates, filename):
    for t in list_of_duplicates:
        if len(t) == 4:
            if dictionary_to_check[t[0]].value > dictionary_to_check[t[1]].value:
                return "\"{}\" is greater than \"{}\" in \"{}\".  Please correct.".format(t[0], t[1], filename)
            if dictionary_to_check[t[0]].value == dictionary_to_check[t[1]].value and dictionary_to_check[t[2]].value != 1:
                return "\"{}\" is equal to \"{}\" but \"{}\" is not 1.  Values are in {}.  Cannot have multiple steps between the same values.  Please correct.".format(t[0], t[1], t[2], filename)
            if dictionary_to_check[t[0]].value != dictionary_to_check[t[1]].value and dictionary_to_check[t[2]].value == 1:
                return "\"{}\" is not equal to \"{}\" but \"{}\" is 1.  Values are in {}.  Cannot have multiple steps between the same values.  Please correct.".format(t[0], t[1], t[2], filename)
            if t[3] and ((dictionary_to_check[t[1]].value - dictionary_to_check[t[0]].value) < (dictionary_to_check[t[2]].value - 1)): #first, last and any integers between.  3-1 =2, so 3-1 steps is valid.  4 steps requires decimals.
                 return "\"{}\" and \"{}\" are close enough that it would require decimals to have provided number of steps. Please adjust minimum, maximum or step number in {}.".format(t[0], t[1], filename)
        elif len(t) == 2:
            if dictionary_to_check[t[0]].value >= dictionary_to_check[t[1]].value:
                return "\"{}\" is greater than or equal to \"{}\" in {}. Please correct".format(t[0], t[1], filename)
    return "Success"


#since we need to do this twice we'll make this function to be run with get_user_defaults  so we can keep all relevant things here
def get_default_values(folder_location, get_user_defaults):
    if get_user_defaults:
        file_name_to_use = os.path.join(folder_location, USER_SETTINGS_FILE)
        dict_to_use = user_settings_and_values
        duplicate_list_to_check = user_settings_min_max
        #need to go a step up in the output folder
        default_settings = {"Output Folder": os.path.join(os.path.dirname(folder_location), "QSRR_Automater_Output"), "Processors_to_Use": multiprocessing.cpu_count()-1} #set the defaults for this one.
    else:
        file_name_to_use = os.path.join(folder_location, PROGRAMER_SETTINGS_FILE)
        dict_to_use = programmer_settings_and_values
        default_settings = {}
    with open(file_name_to_use) as input_file:
        reader = csv.reader(input_file)
        #don't have a header so don't need to drop it
        
        for row in reader:
            if row[0] in dict_to_use.keys():
                dict_to_use[row[0]].check_value(row[1])
                if dict_to_use[row[0]].error:
                    return dict_to_use[row[0]].error
                else:
                    default_settings[row[0]] = dict_to_use[row[0]].value
            else:
                return "The Variable Name \"{}\" is present in the {} file and should not be.  please correct.".format(row[0], file_name_to_use)
        #need to ensure we have all the variables
        missing_variables = [x for x in list(dict_to_use.keys()) if x not in list(default_settings.keys())]
        if missing_variables:
            return "The following variables were not found in {} : {}.  Please correct to proceed.".format(file_name_to_use, missing_variables)
        if get_user_defaults: # no duplicates for programmer
            final_error_check = deal_with_duplicates(dict_to_use, duplicate_list_to_check, file_name_to_use)
            if final_error_check != "Success":
                return final_error_check
        return default_settings