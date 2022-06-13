''' These functions loads datasheet info and organize the data.
pdf_read(file):reads the cable parameters of the pdf-datasheet
file: path directory and file name of the pdf
returns database: the relevant data as pandas.DataFrame
***
cable_locator(cable_layout, data): searches for parameters of a specific cable.
cable_layout: cable configuration (eg. 4 X 2.5) to be searched (string type)
data: DataFrame with the whole data. Usually the database of pdf_read()
returns data.loc[idx]: the relevant data of the specified cable layout
**
Timeline: 21.01.2022: Created by Francinei Vieira 
          02.02.2022: cable_locator() called inside pdf_read(), try/except included 
          02.03.2022: created max_d_locator() and path_finder()
          21.03.2022: try/except replaced by .empty/hasattr()/os._exit()         
'''

import subprocess, sys
try:
    import fitz
except ModuleNotFoundError:
    print('Package PyMuPDF has to be installed\n')
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'pymupdf'])
finally: import fitz              # to read the pdf (PyMuPDF)

import matplotlib.pyplot as plt
import numpy as np
from os import system, _exit
import pandas as pd      # to temp_data organisation
from pathlib import Path # path and file finder

# The colum heading must be given here in the same quantity as in the datasheet
# Names are customisable
col_name = ['series_number','core_type','outer_d','copper_mass','cable_weight'] # LAPP
# col_name = ['AWG','cable_weight','copper_mass','outer_d','core_type','series_number'] # HELUKABEL

def cable_locator(cable_layout, data):
    '''Cable locator function: Given a layout (e.g. 2 X 0.5), the output returns
    the cross-section, cable diameter, n_wires and mass''' 
    cable_layout = str(cable_layout).upper()
    
    if not data[data.core_type == cable_layout].index.empty: 
        idx = data[data.core_type == cable_layout].index[0]
        return data.loc[idx]
    else: 
        print('**** Cable layout NOT found! ****')
        _exit(1) 
    

def max_d_locator(cross_section, data):
    '''Locates the maximum diameter of condutor insulation: Given a cross-section (0.5), 
    the output returns the maxmium diameters of a conductor as per ISO''' 
    if not data[data.cross_sec == cross_section].index.empty:
        idx = data[data.cross_sec == cross_section].index[0]
        return data.loc[idx]
    else:
        print('**** Maximum conductor size was NOT found! ****')
        _exit(1)


def path_finder(input_path=''):
    if input_path == '' :
        if __name__ == '__main__':
            input_path = input('\nEnter the file path of the file: \n \
            Or press ENTER to use the standard datasheet file.\n')
        if input_path == '': input_path = '0'
    try:
        FILE_PATH = Path(input_path).resolve(strict=True)
    except FileNotFoundError: 
        print('**** Using standard file!****')
        # Directory where the .py file is  # ".." means 1 directory above
        SOURCE_DIR = (Path(__file__)/'..').resolve() 
        # Directory to look for the file(s) to be read
        PDF_DIR = (SOURCE_DIR /'..' '/materials').resolve()
        # Specific file to be read
        FILE_PATH = (PDF_DIR / 'LAPP_PRO12ENGB 115CY.PDF').resolve()  # std. file
        print('Opening '+ str(FILE_PATH)+' \n')
    return FILE_PATH


def pdf_read(FILE_PATH = '', cable_layout = '', data_check=False, skip_pdf_read=False):
    '''Read the data on a PDF file and return an Excel table with the aquired data '''
    READ_DATA = (Path(__file__)/'..'/'acquired_data.xlsx').resolve()
    if FILE_PATH == '': FILE_PATH = path_finder()
    # Finding the directory and file 
    try:    # Checking the the file exists in the directory
        if Path(FILE_PATH).resolve(strict=True).is_file(): pass
    except FileNotFoundError:
        print('**** Whoops! No such file or path. Check the name of the file/path. ****\n')
        _exit(1)
    except:  
        print('**** Something else in pdf_read() went wrong... ****\n')
        _exit(1)
    else:
        def check(READ_DATA):
            if READ_DATA.is_file():
                print('\nCheck the read data that will be used in the algorithm.')
                print(f'The file is: {READ_DATA}')
                _ = system(f'start EXCEL.EXE "{str(READ_DATA)}"')
                input("Close the data file and press the <Enter> key to continue...")
                try:    return pd.read_excel(READ_DATA)
                except Exception as e: 
                    print(f'**** {e} ****')
                    _exit(1)
            else: 
                print('****** XLSX not found. Check path or read data again. *******')
                _exit(1)


        if skip_pdf_read is False:
            # The FILE here is read and stored as a multiline-string
            with fitz.open(FILE_PATH) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                text = text.rstrip().split('\n')
            text = pd.Series(text) # converted each line into pd.Series objects
            # print(text)

            temp_data = {x:[] for x in col_name }
            # Read the relevant data a 
            line = 0
            while line < text.size:
                if text[line].isnumeric() and line <= text.size-len(col_name): 
                    for col in range(len(col_name)):
                        temp_data[col_name[col]].append((text[line]))
                        line += 1
                else: line += 1

            # DataFrame conversion. First colunm is kept as string. Others are str or float
            database = pd.DataFrame(temp_data).convert_dtypes().replace(',','.', regex=True) # Obj to strings
            for idx in range(len(database.columns)):
                if database.iloc[:,idx][0].replace('.','').isnumeric(): # Str to float, where needed
                    if database.iloc[:,idx][0] in database.series_number[0]: 
                        pass
                        if hasattr(database, 'AWG'): 
                            if database.iloc[:,idx][0] in database.AWG[0]: pass # skip series_num column
                    else: 
                        try: database.iloc[:,idx] = pd.to_numeric(database.iloc[:,idx], downcast='float')
                        except ValueError: pass
                else: database.iloc[:,idx] = database.iloc[:,idx].str.upper().str.split().str.join(' ')

            # saving the data as an Excel file
            # database = database.sort_values(by='series_number')
            try: database.to_excel(READ_DATA, index=False)
            except PermissionError: print('********* Permission to save the file denied: Check if you ouput file is closed. *********')

            if data_check: database = check(READ_DATA) # Open the data in Excel
            
        elif skip_pdf_read is True:
            if data_check: database = check(READ_DATA)  
            # reading the (edited) data as an Excel file
            else: 
                try: database = pd.read_excel(READ_DATA)
                except Exception as e: 
                    print(f'**** {e} ****')
                    _exit(1)
        else: 
            print(f'**** Error on function {pdf_read.__name__} ****' )
            _exit(1)

        # Look for the desired cable layout
        if cable_layout == '':  cable_layout = input('\nEnter the cable layout desired: ')
        cable = cable_locator(cable_layout, database)

        return cable, database

def wire_processing(cable_data):
    if all(hasattr(cable_data, attr) for attr in col_name):
        n_wires = int(cable_data.core_type.split()[0])
        A_wire  = float(cable_data.core_type.split()[-1])
        r_wire = np.sqrt(A_wire/np.pi)
        cu_mass = float(cable_data.copper_mass)
        cable_weight = float(cable_data.cable_weight)
        outer_d = float(cable_data.outer_d)
        return n_wires, A_wire, r_wire, cu_mass, cable_weight, outer_d
    else: 
        print(f'**** Error on function {wire_processing.__name__} ****')
        return (None,) * 5

