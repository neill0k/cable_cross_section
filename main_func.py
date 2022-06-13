import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd     # If not installed, go to terminal > cmd > and type: py -m pip install --upgrade pandas
from pathlib import Path
# User-created functions
import load_data.load_datasheet as ldt
from optimisation.pack_circles import circle_packing

''' This is the main file of functions for obtaining the 2D cross section of a cable.
INPUTS: DATASHEET (a path to a existing pdf datasheet of a family of cables)
        SEARCH_CABLE_LAYOUT (a text value indicating the desired layout to search and optimize - 
        it must match an existing layout in the datasheet)
OUTPUTS: a xy position and radius of each core (w_n) in the cable.
        a image plot of the respective 2D cross-section with (inner) insulation
'''
 
'''Inputs '''
## Indicate the datasheet file path in the pdf_read or keep it empty to use the standard file
## USE double-backward slash (\\) or a regular slash (/) as path separator
pathfile = r"C:\Users\Documents\Cable_datasheet\YOUR_CABLE_CATALOG.pdf"
DATASHEET = (Path(pathfile.replace('\\', '/'))).resolve() 
DATASHEET = ''   # Empty entry
SEARCH_CABLE_LAYOUT = '4 G 1.5'    # First number is number of cores, the letter is whether has or not GN conductor, last number is the cross-section of the core
## keep it empty to get a prompt message input
# SEARCH_CABLE_LAYOUT = ''
'''Set data_check True if you want to check the data obtained on Excel.
   Set skip_pdf_read True if you want to skip the pdf-data read;
   this avoid overwritting an existing data file '''
data_check=False 
skip_pdf_read=False


'''Execution functions '''
cable, data = ldt.pdf_read(DATASHEET, cable_layout= SEARCH_CABLE_LAYOUT.replace(',','.'), 
        data_check=data_check, skip_pdf_read=skip_pdf_read)
n_wires, A_wire, in_radius, cu_weight, cable_weight, outer_d  = ldt.wire_processing(cable)

## finding core insulation size
DATA_PATH = (Path(__file__) /'..' / 'load_data/max_d_copper_conductors.csv').resolve()
DATA_PATH = ldt.path_finder(DATA_PATH)
max_d_copper = pd.read_csv(DATA_PATH, header=0, delim_whitespace=True, 
                skiprows=[x for x in range(1,7)]).convert_dtypes().astype(float)

if hasattr(ldt.max_d_locator(A_wire, max_d_copper), 'flexible'): r_cond_insulation = ldt.max_d_locator(A_wire, max_d_copper).flexible / 2
else: print('Check your input and try again... Error in ldt.max_d_locator()')

# Select which hyperparemeters to calculate
flags = {'draw_cable_inner_insulation':True,
         'draw_shielding' : True,
         'draw_outer_sheath' : True,
         'copper_density': 8960, 'insulation_density':1350} # material density in kg/m³ for copper/PVC
args = (n_wires, A_wire, in_radius, cu_weight, cable_weight, outer_d, r_cond_insulation, flags)

'''Output '''
# Running the minimization function and getting the plot and data
if None not in (n_wires, in_radius, cu_weight, cable_weight): # test if the variables are valid
    positions = circle_packing(n_cores = n_wires, 
                                inner_radius = in_radius,
                                cond_insulation = r_cond_insulation,
                                var = args
                                )

# saving the data as a xlsx file in the path of this .py file
try: positions.to_excel( (Path(__file__)/'..'/'cable_output.xlsx').resolve(), index=False)
except PermissionError: print('********* Permission to save the file denied: Check if you cable ouput file is closed. *********')

## Display the plot
plt.show()