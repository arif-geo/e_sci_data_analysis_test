""" 
Make a function to plot and save the data at a given timestep.
This can be looped through to make one frame per timestep 
which can then be stitched together into a movie.
"""

def generate_frame(
        timestep : int,
        input_file = "/N/project/obrienta_startup/datasets/ERA5/ds633.0/e5.oper.an.sfc/202106/e5.oper.an.sfc.128_136_tcw.ll025sc.2021060100_2021063023.nc",
        output_dir = "/N/slate/mdaislam/earth_sci_data_analysis_arif/earth_sci_data_analysis/lessons/lesson_09_parallerization/figures"
        ):

        # import the necessary packages
        import os
        import xarray as xr
        import cartopy
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg') # use the 'Agg' backend when running on the cluster
        import cmocean
        import pandas as pd

        # make sure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # load the data file
        ds_in = xr.open_dataset(input_file, chunks=-1)   

        # get the variable at the requested timestep
        tcw = ds_in['TCW'].isel(time=timestep) # isel = index select, tcw = total column water

        # plot the variable on an orthographic projection centered on
        # Bloomington, IN
        clon = -86.5
        clat = 39.2
        projection = cartopy.crs.Orthographic(clon, clat) # Orthographic projection

        # make the empty plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=projection))

        # plot the data
        tcw.plot(
        ax=ax,
        transform=cartopy.crs.PlateCarree(), # always check projection for transform and projection
        cmap=cmocean.cm.haline,
        cbar_kwargs=dict(label=f'Precipitatable water (mm)'),
        vmin=0, # set the min and max values for the colorbar
        vmax=60
        )

        # get the time of the timestep
        time = tcw.time.values

        # convert it to a datetime object
        time = pd.to_datetime(time)

        # add a title with a nicely formatted date
        ax.set_title(time.strftime('%Y-%m-%d %H:%M UTC'), fontsize=16) # strftime = string format time

        # add coastlines
        ax.coastlines(alpha=0.25) # alpha = transparency

        # save the plot
        output_file = os.path.join(output_dir, f'tcw_{timestep:04d}.png')
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)