# cable_cross_section

This project aims for the obtention of the 2D cross section of a cable from the existing data on a datasheet (weight, diameter, number of cores, wire cross section).
It reads the datasheet and uses its data to feed a minimization function (basinhopping-SLSPQ), calculating the optimal radii and coordinates inside a container.

Function main.py: 
INPUTS:

* _DATASHEET_ (a path to a existing pdf datasheet of a family of cables)

* _SEARCH_CABLE_LAYOUT_ (a text value indicating the desired layout to search and optimize - eg. 4 G 1.5,
The 1st number is number of cores, the letter is whether has or not GN conductor, last number is the cross-section of the core. 
It must match an existing layout in the datasheet)
        
OUTPUTS: A table containing a xy position and radius of each core (w_n) in the cable and its insulation.
The radii and thickness of the cable insulation, outer sheath and shielding given are also given. 
A image plot of the respective 2D cross-section with (inner) insulation is displayed.

Some options to check & customize the read PDF data and to modify some of cable's layers are explained in code.

***************
INSTRUCTIONS:
1. Open main_func.py
2. Select the desired input variables and options
3. Run! (needs python +3.7 and some libraries *)

_Observations_:
- This algorithm worked with HELLUKABEL and LAPP datasheets only
- A python interpreter and some libraries (matplotlib, pandas, fitz, numpy, scipy et al.) should be already installed and operative

