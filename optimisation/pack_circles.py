import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform as rand
import pandas as pd
from os import _exit as _exit
from scipy.optimize import minimize, basinhopping
from time import time


def circle_packing(n_cores=None, out_diameter = 2, inner_radius=-1, cond_insulation=-1, var=(None,)*8):
    '''Inputs: number of inner cores/circles (n_cores), 
    opt* outer diameter (out_diameter), internal radius of the cores (inner_radius)
        insulation of the core conductor (cond_insulation)
    Output: DataFrame with xy coordinates and radius of each circle (df)
    The minimization tries to find the maximum radius of the cores 
    for a circle of unitary radius.
    If out_diameter is != 2 the output variables are set proportional to it
    The internal radius > 0, the algorithm seeks the minimum radius of a 
    container that holds all cores inside it. The outer diameter is then ignored.
    '''


    def hyperparameters():
        # cu_mass_diff [kg/km] = (n_cond * conductor's area[mm²]) * copper density [kg/m³] * 1E3[m]/[km] * 1[m²]/1E6[mm²]
        cu_mass_diff    = cu_weight - (n_wires * A_wire * flags.get('copper_density') / 1E3)
        # cond_insul_mass [kg/km] = pi*(r_insul - r_copper)² [mm²]) * insulation density [kg/m³] * 1E3[m]/[km] * 1[m²]/1E6[mm²]
        cond_insul_mass = np.pi*(r_cond_insulation**2 - in_radius**2) * n_wires * flags.get('insulation_density') / 1E3
        insul_mass_diff = cable_weight - cu_weight - cond_insul_mass

        if (flags.get('draw_cable_inner_insulation') or flags.get('draw_outer_sheath')) and not flags.get('draw_shielding'): # (min_radius, thickness)    
            cable_inner_insulation = (out_radius, np.sqrt (insul_mass_diff * 1E3 / (flags.get('insulation_density')*np.pi) + out_radius**2) - out_radius)
            outer_sheath, shielding = ((0, 0),)*2
        elif flags.get('draw_cable_inner_insulation') and flags.get('draw_shielding') and not flags.get('draw_outer_sheath'):
            cable_inner_insulation = (out_radius, np.sqrt (insul_mass_diff * 1E3 / (flags.get('insulation_density')*np.pi)+ out_radius**2) - out_radius) 
            shielding = (sum(cable_inner_insulation), np.sqrt (cu_mass_diff * 1E3 / (flags.get('copper_density')*np.pi)+sum(cable_inner_insulation)**2) - sum(cable_inner_insulation) )
            outer_sheath = ((0,0))
        elif not flags.get('draw_cable_inner_insulation') and flags.get('draw_shielding') and flags.get('draw_outer_sheath'):
            shielding = (out_radius, np.sqrt (cu_mass_diff * 1E3 / (flags.get('copper_density')*np.pi)+out_radius**2) - out_radius )
            outer_sheath = (sum(shielding), np.sqrt (insul_mass_diff * 1E3 / (flags.get('insulation_density')*np.pi)+sum(shielding)**2) - sum(shielding) )
            cable_inner_insulation = ((0,0))
        elif all([val for i, val in enumerate(flags.values())][:3]):
            cable_inner_insulation = (out_radius, np.sqrt (insul_mass_diff * 1E3 * .5 / (flags.get('insulation_density')*np.pi)+out_radius**2) - out_radius )
            shielding = (sum(cable_inner_insulation), np.sqrt (cu_mass_diff * 1E3 / (flags.get('copper_density')*np.pi)+sum(cable_inner_insulation)**2) - sum(cable_inner_insulation))
            outer_sheath = (sum(shielding), np.sqrt (insul_mass_diff * 1E3 * .5 / (flags.get('insulation_density')*np.pi)+sum(shielding)**2) - sum(shielding))
        elif not flags.get('draw_cable_inner_insulation') and flags.get('draw_shielding') and not flags.get('draw_outer_sheath'):
            shielding = (out_radius, np.sqrt (cu_mass_diff * 1E3 / (flags.get('copper_density')*np.pi)+out_radius**2) - out_radius )
            outer_sheath, cable_inner_insulation = ((0, 0),)*2
        elif not all([val for i, val in enumerate(flags.values())][:3]): outer_sheath, cable_inner_insulation, shielding = ((0, 0),) *3
        else: 
            print(f'**** An error occured on function {hyperparameters.__name__}() ****')
            _exit(1)

        return (cable_inner_insulation, shielding, outer_sheath)


    def data_prep(out_radius, inner_radius, cond_insulation, cable_inner_insulation, shielding, outer_sheath):
        '''This function prepares the data output solution, saving as a csv file and plotting 
        the cross-section layout '''
        '''Plots'''
        # Create just a figure and only one subplot
        fig, (ax,ax2) = plt.subplots(2,1, figsize=(6,6),gridspec_kw={'height_ratios': [8, 1]})

        # Remove axes
        ax.axis('off')

        # # Find axis boundaries and circles' data
        circles, lim = find_circles(res.x, out_radius)

        # Data frame
        df = pd.DataFrame({
            "name": ['w'+str(n+1) for n in range(n_cores)]
        })
        if all(var > 0 for var in (inner_radius, cond_insulation)): core_insul_list = []

        # list of labels
        labels = df['name']

        # print circles
        lw = 0.5
        xy_list, core_r_list, circle_list = [ [] for i in range(3)]
        # ax.add_patch(plt.Circle((0,0), out_radius, color='gray',alpha=0.1, linewidth=lw, zorder=5))
        circle_list.append(plt.Circle((0,0), out_radius, color='beige',alpha=0.9, linewidth=2*lw, zorder=3))
        dec = 4     # number of decimal points to record
        for circle, label in zip(circles, labels):
            x, y, r = circle 
            xy_list.append([round(val, dec) for val in circle][0:2])      # create lists for dataFrame
            # If the conductor has insulation, r is the insulation radius; otherwise is only the core radius
            if 'core_insul_list' in locals():  
                core_insul_r = r
                # ax.add_patch(plt.Circle((x, y), core_insul_r, color='teal',alpha=0.4, linewidth=lw, zorder=7))
                circle_list.append(plt.Circle((x, y), core_insul_r, color='green',alpha=0.7, linewidth=lw, zorder=7))
                if label==labels[0]: circle_list[-1].set_label('Core Insulation')
                core_insul_list.append(round(core_insul_r, dec))
            # packing the core radius
            core_r_list.append(round(inner_radius, dec))
            # ax.add_patch(plt.Circle((x, y), inner_radius, color='gold', alpha=1, linewidth=lw, zorder=9))
            circle_list.append(plt.Circle((x, y), inner_radius, color='gold', alpha=1, linewidth=lw, zorder=9))
            if label==labels[0]: circle_list[-1].set_label('Core Conductor')

            ax.annotate(
                label, 
                (x,y ) ,
                va='center',
                ha='center', zorder=10
            )

        # add text info and build up the dataframe (df)
        df = df.assign(x_coord = [row[0] for row in xy_list], y_coord=[row[1] for row in xy_list], core_radius = core_r_list)
        # text = f'Outer radius={out_radius:.3f} mm \nCore radius={inner_radius:.3f} mm'
        if 'core_insul_list' in locals(): 
            df = df.assign(core_insul_radius = core_insul_list)
            # text += f'\nCore Insulation radius={core_insul_r:.3f} mm'
        if outer_sheath is not (None or (0,0)):
            outer_sheath = [round(val, dec) for val in outer_sheath]
            data = pd.DataFrame({'outer_sheath_r': outer_sheath[0], 'outer_sheath_thickness': outer_sheath[1]}, index=['cable'])
            df = pd.concat([df, data], axis=1)
            ax.add_patch(plt.Circle((0,0), sum(outer_sheath), color='royalblue',alpha=0.5, 
            linewidth=lw, zorder=0, label='Outer Sheath Insulation'))
            lim += data.outer_sheath_thickness[0]
        if cable_inner_insulation is not (None or (0,0)):
            cable_inner_insulation = [round(val, dec) for val in cable_inner_insulation]
            data = pd.DataFrame({'cable_inner_insul_r': cable_inner_insulation[0], 'cable_inner_insul_thick': cable_inner_insulation[1]}, index=['cable'])
            df = pd.concat([df, data], axis=1)
            ax.add_patch(plt.Circle((0,0), sum(cable_inner_insulation), color='dimgrey',alpha=0.9, 
            linewidth=lw, zorder=2, label='Insulation'))
            lim += data.cable_inner_insul_thick[0]
        if shielding is not (None or (0,0)):
            shielding = [round(val, dec) for val in shielding]
            data = pd.DataFrame({'shielding_r': shielding[0], 'shielding_thickness': shielding[1]}, index=['cable'])
            df = pd.concat([df, data], axis=1)
            ax.add_patch(plt.Circle((0,0), sum(shielding), color='red',alpha=0.4, linewidth=lw, label='Shielding'))
            ax.set_zorder(0) if outer_sheath == (0,0) else ax.set_zorder(1)
            lim += data.shielding_thickness[0]

        # Add the circles to the figure axes
        [ax.add_patch(patch) for patch in circle_list]
        if (abs(2*lim-outer_d)/outer_d)*100 > 20: 
            print('**** WARNING: The obtained cable diameter has more than 20% difference from the datasheet. *****')

        # ax.text(+lim*1.2, 0, text, ha='left')
        ax.set_aspect(aspect='equal', adjustable='datalim') # improving aspect ratio, adjustable='datalim'
        handles, labels = ax.get_legend_handles_labels()
        ax.autoscale_view()

        # Figure legend
        ax2.axis('off')
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax2.legend(handles, labels, loc='lower center', ncol=2, fontsize='small')      
        ax2.autoscale()
        plt.tight_layout()

        # returning data to main function
        df.name = df.name.replace(np.NaN, 'cable')
        if df.iloc[-1].name == 'cable': df.at['cable','x_coord'], df.at['cable','y_coord'] = (0,0)
        df = df.fillna('-')
        return df


    def distance(x):
        """Non-overlapping: Optimize the distance between 2 cirlces to be >= 2*rin."""
        dist, rad = ([], x[-1])
        x_v = x[:len(x)//2]
        y_v = x[len(x)//2:-1]
        for i in range(n_cores-1):
            for j in range(1,n_cores):
                if j > i: 
                    xk, yk = x_v[i] - x_v[j] , y_v[i] - y_v[j]
                    dist.append( (xk **2 + yk **2) - (2*rad)**2 )
                    # return ( (xk **2 + yk **2) - (2*rad)**2 )
        return dist

    def find_circles(results, r_out=1):
        '''Separate the circles' data and find the axis' boundaries on the figure
        Also calculates the proportion with the outer container, if different from r=1.
        Input: results of minimization (results), outer circle radius (r_out)
        Outputs: circle data x,y,r (circle) and bounds limit (lim)'''
        circle = []
        radius = results[-1]

        for c in range(len(results)//2):
            circle.append([results[c], results[len(results)//2+c], radius])
        circle = np.array(circle) * r_out

        lim = 1.05*max(max((max(abs(circle[:,0])) + max(circle[:,2]),
                max(abs(circle[:,1])) + max(circle[:,2])) for n in circle))
        lim = 1.05 * r_out
        return (circle, lim)

    def max_radii(x):
        '''Maximize the radii of circles, but be inside the container (r=1) 
        x²+y² <= (1-r_in)² '''
        cut = len(x)//2
        r_in = x[-1]
        return [(1-r_in)**2 - x[i]**2 - x[i+cut]**2 for i in range(cut)]

    def min_container(coord, new_r_in):
        old_r_in = coord[-1]
        return new_r_in / old_r_in

    def optimization(find_min_container = False, r_in = None):
        '''Initialization and execution of the miniminization function 
        Returns the full result of minimization (res) and the execution time (optimize_t) '''
        # Calculation of the execution time of this function
        optimize_t = time()

        # Initial guess
        x0 = [*rand(low=0.2, size=2*n_cores+1)]   # x1,x2,..xn,y1,y2,...,yn, rin

        # Find maximun radius of inner circles
        # Constraints - refer to other functions, e.g. distance, max_radii
        cons=({'type': 'ineq', 'fun': distance},
            {'type': 'ineq', 'fun': max_radii},
            {'type': 'ineq', 'fun': lambda x: np.array([1-abs(x[i]) for i in range(len(x)-1)])}, # -1 < xy < 1
            {'type': 'ineq', 'fun': lambda x: np.array([1 - abs(x[-1])])}, # 0 < radius < 1
        ) # the inequations must fulfill >= 0
        # res = minimize(lambda x:-x[-1], x0, method='SLSQP', constraints=cons, options={'gtol':1E-9, 'disp':1}) # alternative
        argument = dict(method='SLSQP', constraints=cons)
        res = basinhopping(lambda x:-x[-1], x0, minimizer_kwargs=argument)
        out_radius = 1

        if find_min_container: # Find minimum outer container r_out
            if r_in is None: print('Internal radius was NOT given!')
            else:
                out_radius = min_container(res.x, r_in)
        optimize_t = time() - optimize_t
        return res, out_radius, optimize_t




    # Core insulation check
    if inner_radius > 0 and cond_insulation > 0: core_radius = cond_insulation # r = core + insulation thickness
    else: core_radius = inner_radius

    # Execution of the minimization
    if inner_radius > 0 :
        if out_diameter != 2:  print('********* Conflituous condition! The outer diameter will be ignored *********' )  # inner_r > 0 and out_d != 2 is a conflituous condition         
        print("********* Finding the outer circle's minimun radius. *********\n" )
        res, out_radius, exec_time = optimization(find_min_container=True, r_in=core_radius)
    elif inner_radius <= 0 :  
        print("********* Finding the inner circles' maximum radius in a container r=1. *********" )
        res, out_radius, exec_time = optimization()

    # Atribbute check and error handling
    if hasattr(res, 'success'): 
        message = getattr(res, 'message')
        success = getattr(res, 'success')
    elif hasattr(res.lowest_optimization_result, 'success'): 
        message = getattr(res.lowest_optimization_result, 'message')
        success = getattr(res.lowest_optimization_result, 'success')
    if success: 
        print(f'SUCCESS!!! {res.message[0]} in {exec_time:.2f} s.\n ')
        
        # Calculate extra parameters (hyperparameters)
        n_wires, A_wire, in_radius, cu_weight, cable_weight, outer_d, r_cond_insulation, flags = [i for i in var]
        # add data to df
        df = data_prep(out_radius, inner_radius, cond_insulation, *hyperparameters())

        return df

    else:
        print('**** The optimization did not converge! ****')
        _exit(1)
