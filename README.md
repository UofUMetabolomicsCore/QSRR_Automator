# QSRR Automator
Easy to use Automation software for creation of Retention Time prediction machine learning models. 
QSRR Automator was written in and tested on the Windows operating system. There may be bugs if run on Apple or Linux operating systems.
The code is provided here for transparency and the ability for the user to modify if desired.  An .exe version is provided at https://github.com/UofUMetabolomicsCore/QSRR_Automator/releases/tag/v1_exe.

The code requires the following dependencies:
-rdkit
-mordred
-pyqt5
-scikit-learn

There are other dependencies which likely already installed in base python installations or as prerequisites to the other dependencies:
-matplotlib
-scipy
-numpy
-pandas

it is advised to use anaconda due to the difficulty in installing rdkit, since "conda install -c rdkit rdkit" or other commands from https://anaconda.org/rdkit/rdkit.  Instructions for if conda installation fails or you wish to use a different python environment can be found here: https://www.rdkit.org/docs/Install.html.  All other dependencies are available as standard pip installations

To use, download all files into the same folder and run the "\_\_main\_\_.py".  In Windows this should automatically occur with a "python {full path of code folder} command in the command line. QSRR Automator does require the ability to write folders and files and its default location for writing is immediately adjacent to the folder containing the code files. It is advisable to place the QSRR Automater folder in an easy to access location that does not require administrator access to write to.

QSRR Automater will open a GUI which should hopefully be relatively self-explanatory.

For further details on the various GUI elements, settings, and outputs please read "QSRR Automator Instruction Manual.docx", provided along with the other files.

