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

#need to test the display without waiting for Random Forest or SVR to complete

import os, sys
#import numpy as np
#import pandas as pd


from PyQt5 import QtCore, QtWidgets, uic
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas 


import settings_from_model_graph as sfmg


#from matplotlib.backends.backend_qt4agg import FigrueCanvasQTAgg as FigureCanvas
temp_loc = os.path.dirname(os.path.abspath(__file__))

graph_filename = "Successful_Graph.pdf"

form_class = uic.loadUiType(os.path.join(temp_loc,"Model_Training_Viewer.ui"))[0]
class display_training_results(QtWidgets.QDialog, form_class):
    def __init__(self, final_model = None, location = None, settings = None, settings_from_last_run = None, original_settings = None, number_of_features = None):
        QtWidgets.QMainWindow.__init__(self)
        self.setWindowTitle("Model Examination")
        self.setupUi(self)
        self.location = os.path.join(location, graph_filename)
        
        #need to set up the settings adjustment settings
        self.settings = settings
        self.last_save_settings = settings_from_last_run
        self.original_settings = original_settings
        self.number_of_features= number_of_features
        
        self.actual_times = final_model.y_values
        self.predicted_times = final_model.predicted_y
        
        self.value_mean_cross_val_r2 = final_model.r2_mean
        self.value_median_cross_val_r2 = final_model.r2_median
        self.mean_time_error_time_units = final_model.mae_mean
        self.final_model_mean_r2_score = final_model.final_r2_score
        self.final_model_mean_mae_score = final_model.final_mae
        
        
        self.max = 0
        self.min =10
        for temp_list in [self.actual_times, self.predicted_times]:
            if self.max < max(temp_list): self.max = int(max(temp_list) + .5)
            if self.min > min(temp_list):self.min = int(min(temp_list))
        self.ideal_fit = lambda x: x
        self.return_value = "Not Selected"
        self.adjust_settings_button.clicked.connect(self.change_settings_for_next_model)
        self.redo_button.clicked.connect(self.bad_model)
        self.accept_button.clicked.connect(self.good_model)
        self.graph()
        
        
                
    def graph(self):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111) #standard matplotlib, linked to the canvas object
        self.ax.plot(self.actual_times, self.predicted_times, 'b*', markersize = 10, label = "Final Model Data")
        self.ax.plot([self.min, self.max], [self.min, self.max], 'k-', label = "Unity Line (Ideal)")
        plt.title("How good does the Model do?", fontsize = 18)
        plt.xlabel("Actual RT", fontsize = 16)
        plt.ylabel("Predicted RT", fontsize = 16)
        
        self.ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

        self.ax.legend(bbox_to_anchor = (.15, 1.1))
        #based on https://stackoverflow.com/questions/53157230/embed-a-matplotlib-plot-in-a-pyqt5-gui first answer accessed 1/25/19
        self.plotWidget = FigureCanvas(self.figure)
        layout = QtWidgets.QVBoxLayout(self.graph_view)
        layout.addWidget(self.plotWidget)
        
        #we need to limit the decimal points since there is no need for tons of them and they clutter the display quite a bit.
        self.mean_cv_r2.setText("Mean Training R2 = {:.3f}".format(self.value_mean_cross_val_r2))
        self.median_cv_r2.setText("Median Training R2 = {:.3f}".format(self.value_median_cross_val_r2))
        self.mean_cv_absolute.setText("Expected Absolute Error of Predictions = {:.3f}".format(self.mean_time_error_time_units))
        self.final_r2_score.setText("Final Model R2 Score = {:.3f}".format(self.final_model_mean_r2_score))
        self.final_absolute_error.setText("Final Model Mean Absolute Error = {:.3f}".format(self.final_model_mean_mae_score))
        
        
        #add values from the machine learning analysis (will need to check if there are the same
    def bad_model(self):
        self.return_value = "Redo Calculation"
        self.close()
    def good_model(self):
        self.return_value = "Accept Model"
        try:
            self.ax.figure.savefig(self.location,bbox_inches = 'tight')
        except PermissionError:
            QtWidgets.QMessageBox.information(self, "Error", "{} is already open. close to accept and save the graph".format(self.location))
            return
        self.close()
        
    def change_settings_for_next_model(self):
        self.set_menu = sfmg.Settings_Menu(self, self.settings, self.last_save_settings, self.original_settings, self.number_of_features)
        self.set_menu.show()
    
    
    def closeEvent(self, event):
        if self.return_value != "Not Selected":
            event.accept()
        else:
            are_you_sure = QtWidgets.QMessageBox.question(self, "Exit", "You will be returned to the main menu with no model saved and no further calculations.  Are you sure?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if are_you_sure == QtWidgets.QMessageBox.Yes:
                event.accept()
            else:
                event.ignore() #functioning
         
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    interaction_gui = display_training_results()
    interaction_gui.show()
    sys.exit(app.exec_())