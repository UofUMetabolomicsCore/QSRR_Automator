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
import os
import pandas as pd
#import time
def make_folder(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        
def check_file_exists(full_path):
    """
    if not os.path.isfile(full_path):
        print (full_path)
        return "File Does Not Exist"
    """
    try:
        test = open(full_path, 'w')
        test.close()
        return "File Is Available"
    except PermissionError:
        return "{} already open in another program. Close and try again.".format(full_path)

#want to check if a file exists and then write.
#since we have several instances this will allow us to for loop over them easily
def write_single_file(file_path,thing_to_write, use_index, use_excel):
    results = check_file_exists(file_path)
    if results != "File Is Available":
        return results
    if use_excel:
        if use_index:
            thing_to_write.to_excel(file_path)
        else: 
            thing_to_write.to_excel(file_path, index = False)
    else:
        if use_index:
            if isinstance(thing_to_write, pd.Series):
                thing_to_write.to_csv(file_path, header = True)
            else:
                thing_to_write.to_csv(file_path) 
        else: 
            if isinstance(thing_to_write, pd.Series):
                thing_to_write.to_csv(file_path, header = True, index = False)
            else:
                thing_to_write.to_csv(file_path, index = False)
    return "Success"
    
def deal_with_writer_object(excel_writer, fatal_error = False):
    if not fatal_error:
        try:
            excel_writer.save()
        except IndexError:
            pd.DataFrame({"Empty":range(3)}).to_excel(excel_writer)#if the excel_writer is empty it can't close. this just adds a blank sheet so we can close appropriately and then remove the empty file  
            excel_writer.save()
            fatal_error = True #no need to keep the file since it is empty
        except PermissionError:
            return "Permission Error" # no need to close here since it is already open is something else (thus the Permission Error)
    else:
        #if the excel_writer is empty it can't close. this just adds a blank sheet so we can close appropriately if we close before adding s
        pd.DataFrame({"Empty":range(3)}).to_excel(excel_writer)
    excel_writer.close()
    #delete the error filled file.  can comment out if needed, but we aren't saving it anyway.
    if fatal_error:
        os.remove(excel_writer.path)
        
def make_series_for_saving(dictionary):
    series_to_write = pd.Series(dictionary)
    series_to_write.index.name = "Setting"
    series_to_write.name = "Value"
    return series_to_write