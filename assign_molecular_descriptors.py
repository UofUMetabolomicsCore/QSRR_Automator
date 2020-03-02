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
For creating the model or using it we need the same input save for the RT column, so we may as well use the same function.
We'll be stealing a fair bit from the current button two, but here are the goals
    import the settings, a boolean governing which buttton we are using, output folder, and excel file if relevant
    also import a .csv or .xlsx file (ask the user for it in the main)
    use boolean to check the file.  
        /if missing necessary headers just reject it (they will have a template, no excuse for messing this up.)
        /if it has RT when should not, reject it (assume we're using the template option so they selected the wrong file.  if they want it in there they can change the name)
        /for the following settings I use 3.  this should be a setting.  if we're using "Use Feature Selection", this should be "Number of Features Desired".  if not, have a minimum in the programmer settings
        -check for extra columns
            - if training make sure we have 3+  and then ask if they want to keep these values
                -if less than 3 complain to the user
            -if rt assignment, check if these include ALL of the features needed in the model 
                -if so can save a trimmed version
                -if not complain to the user
        -if have needed columns (rt assignment) or the user wishes to use them return the packaged pandas object with no further manipulation
            -do check in the same way as mordred columns (not too many are the same, no nans, correlations, etc.)
        -if wants to trim the values or does not have the appropriate values, take the needed values and run mordred
            -keep all existing filters for the mordred stuff
        -in either case check that the final amount is greater than whatever the minimum required columns is
            -if fails complain to the user
            
    bug check all of the above (/ coded, x complete)
"""

import os
import numpy as np
from collections import Counter

import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors

import generic_functions as gf

DROP_3D = True
compound_column_name = "Compound Name"
smile_column_name = "SMILES"
rt_column_name = "RT"
grouping_data_filename = "Mordred_Descriptor_Key.csv"

np.seterr(all='raise')

#this calculates the fraction of valid values in a 
def fraction_of_nans (list_to_analyze):
    nan_counter = 0
    for l in list_to_analyze:
        if np.isnan(l): 
            nan_counter +=1
    return nan_counter/len(list_to_analyze)

#returns the fraction of a list that contains the most frequent entry (looking for lists that are all one thing)
def check_for_too_much_similarity(list_to_analyze):
    counter_dict = Counter(list_to_analyze) #should return a dictionary with the elements of the list as keys and the number of their occurances
    return max(counter_dict.values())/len(list_to_analyze)#find the biggest count (values are counts in a dict from Counter) and see what fraction of the list it is.

def make_list_to_drop(input_series, lower_is_good):
     drop_columns = []
     for i in input_series.index:
         if input_series[i] >=lower_is_good:
             drop_columns.append(i)
     return drop_columns
     
def determine_descriptors_from_mordred(smile_series):
    smile_series = smile_series.drop_duplicates() #no need for duplicates they just take time, we can merge by smiles later to sort it out 
    calc = Calculator(descriptors, ignore_3D = True) # create a calculation object (ignore_3D is the default, just a reminder it is there)
    molecule_objects = []
    bad_smiles = []
    for smile in smile_series:
        if type(smile) != str:
            bad_smiles.append(smile)
            continue
        to_check = Chem.MolFromSmiles(smile) #a "SMILES Parse Error" does not trigger 
        if to_check:
            molecule_objects.append(to_check)
        else:
            bad_smiles.append(smile)
    if bad_smiles:
        return bad_smiles, False
    descriptor_dataframe = calc.pandas(molecule_objects) #so long as all smiles are valid this should be fine
    #need to merge smiles with the descriptor_dataframe
    descriptor_dataframe= descriptor_dataframe.set_index(smile_series)
    return descriptor_dataframe, True
 
def filter_out_correlations(main_df, descriptor_df, user_settings, programmer_settings):
    #for now we have not determined how to deal with 3D structural descriptors in mordred.  if you do, here's were it goes
    if DROP_3D:
        descriptor_df = descriptor_df[descriptor_df["Dimension"] != "3D"]
    #$shouldn't have issues with boolean columns, or compounds we don't need to analyze (rt, compouns, smiles). if we do, deal with that here
    columns_to_drop = []
    current_columns = list(main_df.columns)
    for descriptor_family_name, descriptor_family_data in descriptor_df.groupby("Family"):
        #no need to bother is there is only one mention of the family
        if len(descriptor_family_data) ==1:
            continue
        relevant_columns = list(descriptor_family_data["Descriptor"])
        temp_dataframe = main_df.loc[:,main_df.columns.isin(relevant_columns)] #not sure if this will work if the descriptors are not present isin suggested by https://stackoverflow.com/questions/43537166/select-columns-from-dataframe-on-condition-they-exist first answer 10/31/18
    
        correlation_dataframe = temp_dataframe.corr() #this will create a columns by columns matrix with pearson coefficients
        length_to_consider = len(correlation_dataframe.columns)
        temp_bad_columns = []
        #loop should keep the first column in the correlation (have to choose one, so first is as good as any other
        for i in range(length_to_consider):
            for j in range(i+1, length_to_consider):
                if correlation_dataframe.iloc[i,j] > user_settings["Correlation Coefficient (Pearson)"]:
                    temp_bad_columns.append(list(correlation_dataframe.columns)[j])
        columns_to_drop.extend(list(set(temp_bad_columns))) #may as well not store tons of duplicates
    columns_to_keep = [c for c in current_columns if c not in columns_to_drop]        
    return main_df[columns_to_keep]  

#now to actually filter the problems
#again based on a previous work so problems should be minimal.
def filter_descriptors(numerical_df, user_settings, programmer_settings, input_location, excel_object = None):
    max_replicate_descriptors_observed = numerical_df.apply(check_for_too_much_similarity, axis = 0)
    #no need to check the similarity of numbers with only one sample
    if len(numerical_df.index) > 1: 
        columns_to_drop = make_list_to_drop(max_replicate_descriptors_observed, user_settings["Allowed same value per descriptor"])
        numerical_df = numerical_df.drop(columns_to_drop, axis =1)
        if user_settings["Excel Writing"]:
            numerical_df.to_excel(excel_object, "Replicates Removed")
        else:
            write_problem = gf.write_single_file(os.path.join(user_settings["Output Folder"], "Model building replicates removed.csv"),numerical_df, True, False)
            if write_problem != "Success":
                return write_problem
    #now we need to deal with bad samples of various types
    bad_samples = numerical_df.apply(fraction_of_nans, axis = 1) #leave out the compound name column (it's easier on the functions if we don't have a named index because of the posibility of duplicat compounds with separate rts
    rows_to_drop = make_list_to_drop(bad_samples, user_settings["NaNs allowed per sample"])
    numerical_df = numerical_df.drop(rows_to_drop, axis = 0)
    if len(numerical_df.index) ==0:
        return "All samples were removed for failing various filters molecular descriptors.  choose a different file or adjust settings to correct" 
    if user_settings["Excel Writing"]:
        numerical_df.to_excel(excel_object, "Bad Samples Removed")
    else:
        write_problem = gf.write_single_file(os.path.join(user_settings["Output Folder"], "Model building bad samples removed.csv"),numerical_df, True, False)
        if write_problem != "Success":
            return write_problem
    #drop all nans. unsure of effect on machine learning, and most descriptors don't have one or two nans, they have several. so that is important to drop
    #$consider if we are doing something like HILIC on multiple compound classes, may need to keep (would likely need a placeholder value of some descripion)
    filtered_df = numerical_df.dropna(axis = 1) 
    #minimum_number_of_features  is based on the maximum that the user is asking for
    if user_settings["Feature Selection Method"] == "Manual" : minimum_number_of_features = user_settings["Target Number Features for Manual"]
    else: minimum_number_of_features = programmer_settings["Minimum Machine Learning Descriptors"]
    if len (filtered_df.columns) < minimum_number_of_features:
        return "Insufficient descriptors remaining after filtering.  Please adjust filters or use different SMILES"
        
    if user_settings["Excel Writing"]:
        filtered_df.to_excel(excel_object, "Final Descriptors")
    else:
        write_problem = gf.write_single_file(os.path.join(user_settings["Output Folder"], "Model building Final Descriptors.csv"),filtered_df, True, False)
        if write_problem != "Success":
            return write_problem
    descriptor_grouping_df = pd.read_csv(os.path.join(input_location,grouping_data_filename))
    trimmed_dataframe = filter_out_correlations(filtered_df, descriptor_grouping_df, user_settings, programmer_settings)
    return trimmed_dataframe
                       
#i have written this previously so hopefully problems are minimal.    
def actually_add_descriptors(input_data):
    #now are sufe there are no blanks and all the values are good, we can start things like numberical coersion, writing stuff out and mordred
    #first we need to actually perform mordred search
    descriptor_dataframe, proceed = determine_descriptors_from_mordred(input_data[smile_column_name])
    if not proceed:
        return "These SMILES were invalid: {}.  Please correct or remove them and try again.".format(", ".join(descriptor_dataframe))
    numerical_df = descriptor_dataframe.apply(pd.to_numeric, args = ['coerce']) #should end up with numbers, nans and booleans.
    return numerical_df




#this is an initial read in of the file, ensure the columns are correct and so on. data is returned to the driving function in the main for further decisions.
def initial_preparation_machine_learning_input(input_filename, user_settings, programmer_settings, need_rt, current_descriptors_names, reading_in_auto = False):
    #read in data
    try:
        if input_filename[-4:] == ".csv":
            input_data = pd.read_csv(input_filename)
        elif input_filename[-5:] == ".xlsx":
            input_data = pd.read_excel(input_filename)
        else:
            return "Do not recognize file extension", "" #this should be unnecessary but the computational burden should be basically nothing to account for other changes in the code or prevent problems I didn't anticipate
    except UnicodeDecodeError:
        return "There is a non-parsable character in your input data.  Try finding or replacing this character or saving the input file as a different input filetype.", ""
    #check for needed columns
    #functional
    for name in [compound_column_name, smile_column_name]:
        if name not in input_data.columns:
            return "{} not in header.  This is case sensitive.  Please correct and try again".format(name), ""
    #functional 
    if need_rt and rt_column_name not in input_data.columns:
        return "{} not in header. This is case sensitive. Please correct and try again".format(rt_column_name), ""
    
    if not need_rt and rt_column_name in input_data.columns:
        return "{} in header when it should not be.  Please correct and try again".format(rt_column_name), ""
    #functional
    if input_data.empty:
        return "No data in input file.  Please correct and try again.", ""
    #make sure data in the file is present.  no blanks or nans.
    last_good_index = input_data.last_valid_index()
    if last_good_index == None:
        return "No good data in your input file.  Please try again.", ""
    input_data = input_data[:last_good_index+1] #trim empty rows from the bottom if excel adds any
    #functional for blanks
    #need to make sure we have enough samples to reasonably analyze (3 samples for example will error out with sklearn
    if len(input_data.index) < programmer_settings["Minimum Number of Samples Allowed"] and need_rt: #minimum is for sklearn assignment.  if we don't need rt, so we are predicting, we don't need a limit since predicting less than 50 will often be needed
        return "There are {} samples in the input file, when a minimum of {} are required.".format(len(input_data.index), programmer_settings["Minimum Number of Samples Allowed"]), ""
    #functional for blanks
    if input_data.isnull().values.any():
        return "Your input file contains missing values.  Please correct and try again.", ""
    #need to ensure all names have unique smiles
    instances_of_each_name = Counter(input_data[compound_column_name])
    potential_issues = [k for k in instances_of_each_name.keys() if instances_of_each_name[k] > 1]
    for p in potential_issues:
        #names are the same so we just need to check the smiles
        small_df = input_data[input_data[compound_column_name] == p]
        smiles = list(small_df[smile_column_name])
        if len(set(smiles)) != 1: #we should only have one smile for each name, so if there are more than one mile, 
            return "The compound {} has more than one SMILES value associated with it. Please correct.".format(p), ""
    #if we have rt we need to deal with ensuring that all rt values are greater than 0 and void volume
    if need_rt:
        #negative values don't make sense (and if it's before an official start time that should be trimmed as it is likely minimally predictable)
        if any(input_data[rt_column_name] < 0):
            return "There are negative values in your retention time column.  Please correct.", ""
        input_data = input_data[input_data[rt_column_name] >= user_settings["Void volume cutoff"]]
        if input_data.empty:
            return "None of your samples were greater than your void volume cutoff. Please adjust your data or settings and try again.", ""
        if len(input_data.index) < programmer_settings["Minimum Number of Samples Allowed"]:
            return "There are {} samples with retention times greater than the void volume cutoff time in the input file, when a minimum of {} are required.".format(len(input_data.index), programmer_settings["Minimum Number of Samples Allowed"]), ""
    #now that we are sure the data is good, we need  replace the columns with the numerical values (largely this is to do th
    #functional
    if user_settings["Feature Selection Method"] == "Manual":
        min_required_descriptor_columns = user_settings["Target Number Features for Manual"]
    else:
        min_required_descriptor_columns = programmer_settings["Minimum Machine Learning Descriptors"]
    #determine if there are extra columns
    #assume all needed columns are present since we just checked
    if need_rt: current_descriptors = len(input_data.columns)-3
    else: current_descriptors = len(input_data.columns)-2
    #functional
    if current_descriptors == 0:
        need_mordred = True
        
    elif current_descriptors >= min_required_descriptor_columns:
        need_mordred = False #at this point this is tentative.  we will ask in the main if we need to keep these or reassign
        #we need to get rid of non numerical (or boolean, but that can be numerical) data
        #current idea: ignore columns that we know are there (since we checked they are there already we can cut them out and merge back later if needed.
        #then check for nulls (Note: it is very possible to use strings for the machine learning of course, but mordred only does numerical or boolean, and forcing the user to go numerical is less bother than allowing it, then allowing for every hare-brained thing they might do with strings)
        numerical_columns = list(input_data.columns)
        if need_rt:numerical_columns = [n for n in numerical_columns if n not in [compound_column_name, smile_column_name, rt_column_name]]
        else: numerical_columns = [n for n in numerical_columns if n not in [compound_column_name, smile_column_name]]
        numerical_input_data = input_data[numerical_columns]
        numerical_input_data = numerical_input_data.apply(pd.to_numeric, args = ['coerce'])
        if numerical_input_data.isnull().values.any():
            return "Your file seems to contain descriptor columns, but some of the data they contain is non-numeric.  please correct and try again", ""
        if need_rt: input_data = pd.concat([input_data[[compound_column_name, rt_column_name, smile_column_name]], numerical_input_data], axis =1 )
        else: input_data = pd.concat([input_data[[compound_column_name, smile_column_name]], numerical_input_data], axis =1 )
        #we do need to check that all needed features are present
        if not need_rt: #only the case if we are using a model for prediction (only case where current_model with exist
            #if the feature selection is automatic, it requires chosen features only.  since this is different from none and manual we should help out here
            if reading_in_auto == "Automatic":
                needed_columns = [compound_column_name, smile_column_name]
                needed_columns.extend(current_descriptors_names)
                for n in needed_columns:
                    if n not in input_data.columns: print(n)
                try:
                    input_data = input_data[needed_columns] #need to trim to just the automatic columns
                except:
                    return "The chosen features for your model were not present in your input file.  Please correct", ""
            
            #need to get the column names
            check_features = list(input_data.columns)
            check_features.remove(compound_column_name)
            check_features.remove(smile_column_name)
            if not set(current_descriptors_names) == set(check_features): #if chosen features is not a subset (does include full equality) there is an issue and model will not fit it
                return "Your file contains columns that are not in the template, or do not correspond to the needed features for the model you are using.  This requires all features used in the initial training for \"None\" or \"Manual\" feature selection or the needed features for \"Automatic\" features selection.  This is case sensitive. Please correct and try again", ""
    #functional
    else:
        return "Your file seems to contain descriptor columns, but there are insufficient numbers of them for your settings.  please correct the file or the settings", ""
    
    
    return need_mordred, input_data
    

    