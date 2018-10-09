# -*- coding: utf-8 -*-
"""
@author: Daniel Reyes Lastiri
Templates for plots

To customize plots see:
http://matplotlib.org/users/customizing.html#matplotlibrc-sample
For mathematical expressions see:
http://matplotlib.org/users/mathtext.html
"""
import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
import matplotlib.cm as cm
from matplotlib import colors, gridspec
from matplotlib.ticker import AutoLocator, AutoMinorLocator
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches

# To locate folder of styles
#from matplotlib.style.core import USER_LIBRARY_PATHS

### TODO
# pass fig to functions, that way size can be given as function parameter
# subtract amount of line-marker pairs from number of colors in colormap collin


# Set the general font properties for the plot
#rcParams['font.monospace'] = 'Fixed' #monospace fonts help improving kerning
#rcParams['font.size'] = 14
#rcParams['legend.fontsize'] = 14
#rcParams['xtick.labelsize'] = 18
#rcParams['ytick.labelsize'] = 18
#rcParams['axes.labelsize'] = 18
#rcParams['mathtext.default'] = 'regular'

# To set locators of ticks
# http://matplotlib.org/api/ticker_api.html#tick-formatting

# For a complete list of colormaps
# colmaplist = plt.colormaps()

def plot_lines(ax,x_list,y_list,xlabel=r'$x$',ylabel=r'$y$',title=None,
               linestyle_list=None,marker_list=None,color_list=None,
               label_list=None, legend_loc='upper center',
               xpad=0.5, ypad=0.2, legend_cols=2):
    # Load style file
    plt.style.reload_library()
    if 'paper' in plt.style.available: plt.style.use('paper')
    else: plt.style.use('ggplot')
    # Transform single array of x into list of arrays with equal values
    if not type(x_list) == list: x_list = [x_list,]*len(y_list)
    if not type(y_list) == list: y_list = [y_list,]
    if not type(label_list) == list: label_list = [label_list,]*len(y_list)
    # Transform None and False *kwargs into list with len = len(y_list)
    # in order to allow for iteration
    if linestyle_list == None: linestyle_list = ['-',] * len(y_list)
    if marker_list == None: marker_list = [None,] * len(y_list)
#    if label_list == None: 
#        label_list = ['y{}'.format(i) for i, v in enumerate(y_list)]
    # Colors
    if color_list:
        ax.set_prop_cycle(color=color_list)
    # Plot
    #ax = plt.subplot2grid((4,1), (0, 0), rowspan=3)
    for i,yi in enumerate(y_list):
        ax.plot(x_list[i],yi,linestyle=linestyle_list[i],
                marker=marker_list[i],label=label_list[i]
                ) 
    # Title
    if title: ax.set_title(title, y=1.08)        
        #fontForTitle = {'fontname':'Century'}
    # Axes labels from the names given in xlabel and ylabel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)   
    # Minor tick location, dividing major in 2
    ax.autoscale()
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))    
    # Set the grid with transparency alpha
    ax.yaxis.grid(True, which='major')
    ax.xaxis.grid(True, which='major')    
    # Legend
    #ax.legend(bbox_to_anchor=(1.02,0.5), loc='center left')
    if label_list:
        ax.legend(bbox_to_anchor=(xpad,-ypad),
                  loc=legend_loc, ncol=legend_cols,
                  columnspacing=1, handletextpad=0.5,
                  prop={'size':9})
    return ax

def plot_scatter(ax,data_list,xlabel=r'$x$',ylabel=r'$y$',title=None, ms=7,
                 marker_list=['o',],label_list=None,spheres=None,
                 legend_pad=0.2,legend_cols=2):
    # Load style file
    plt.style.reload_library()
    if 'paper' in plt.style.available: plt.style.use('paper')
    else: plt.style.use('ggplot')
    # Transform single array of x into list of arrays with equal values
    if not type(data_list) == list: data_list = [data_list,]
    # Transform None and False *kwargs into list with len = len(y_list)
    # in order to allow for iteration
    if label_list == None:
        label_list = ['y{}'.format(i) for i, v in enumerate(data_list)]
    # Plot
    for i,data in enumerate(data_list):
        ax.scatter(data[:,0],data[:,1],
                   marker=marker_list[i],label=label_list[i],s=ms)  
    # Title
    if title: ax.set_title(title, y=1.08)
        #fontForTitle = {'fontname':'Century'}
    # Show top and right axes (and ticks) (my 'paper' style removes them)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    # Axes labels from the names given in xlabel and ylabel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)   
    # Minor tick location, dividing major in 2
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))    
    # Set the grid with transparency alpha
    ax.yaxis.grid(True, which='major')
    ax.xaxis.grid(True, which='major')
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Add spheres
    if spheres:
        for i,center in enumerate(spheres['centers']):
            radius = spheres['radii'][i]
            circleplt = plt.Circle(center,radius,color='#d490c6',alpha=0.4)
            ax.add_artist(circleplt)
        # Legend w/spheres
        sphere_patch = mlines.Line2D([],[], linewidth=0, marker='o',
            mfc='#d490c6', ms=10, mec = '#d490c6',alpha=0.5)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(sphere_patch)
        labels.append(r'$FPS$')

    ax.legend(handles,labels, bbox_to_anchor=(0.5,-legend_pad), loc='upper center',
              ncol=legend_cols, columnspacing=0.2, handletextpad=0.05)
    return ax
        
def plot_uncertainty(ax,xdata,ydata,errbars,ymin,ymax,
                     errbelowmin,errabovemax,shortabovemin,shortbelowmax,
                     xlabel=r'$x$',ylabel=r'$y$',
                     cbounds='#2d8ce5',cuncert='#af0000',
                     line=None,label_line=None,legend_pad=0.2):
    plt.style.reload_library()
    if 'paper' in plt.style.available: plt.style.use('paper')
    else: plt.style.use('ggplot')
    # Plot error bars and uncertainty
    ax.errorbar(xdata,ydata, yerr=errbars, fmt='none', elinewidth=2,
                capsize=4.8, capthick=2, ecolor=cbounds)
    ax.errorbar(xdata,ymax, errabovemax, fmt='none', elinewidth=1,
                capsize=3.2, capthick = 1, ecolor=cuncert)
    ax.errorbar(xdata,ymin, errbelowmin, fmt='none', elinewidth=1,
                capsize=3.2, capthick = 1, ecolor=cuncert)
    ax.errorbar(xdata,ymin, shortabovemin, fmt='none', elinewidth=1,
                capsize=3.2, capthick = 1, ecolor=cuncert)
    ax.errorbar(xdata,ymax, shortbelowmax, fmt='none', elinewidth=1,
                capsize=3.2, capthick = 1, ecolor=cuncert)
    # Axes labels from the names given in xlabel and ylabel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)  
    # Minor tick location, dividing major in 2
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))    
    # Set the grid with transparency alpha
    ax.yaxis.grid(True, which='major')
    ax.xaxis.grid(True, which='major') 
    # Legend
    bounds_bars = mlines.Line2D([], [],linewidth=2,color=cbounds)#, label='Bounds')
    uncert_bars = mlines.Line2D([], [], color=cuncert)#, label=r'$OE$ and $UE$')
    ax.legend(handles=[bounds_bars,uncert_bars],
              labels=['Bounds',r'$OE$ and $UE$'],
              bbox_to_anchor=(0.5,-legend_pad),
              loc='upper center',ncol=2)
    # Lines
    if line:
        ax.plot(line[0],line[1],'--')
        original = mlines.Line2D([], [],linewidth=1.2,ls='--',
                                 color='k')#,label=label_line)
        ax.legend(handles=[bounds_bars,uncert_bars,original],
                  labels=['Bounds',r'$OE$ & $UE$',label_line],
                  bbox_to_anchor=(0.5,-legend_pad), loc='upper center', ncol=3)
    return ax

def plot_scatter3d(ax3d,data_list,
                   xlabel=r'$x$',ylabel=r'$y$',zlabel=r'$z$',
                   marker_list=None, label_list=None,
                   axislim=[(-0.5,1.5),]*3,spheres=None):
    # Transform None and False *kwargs into list with len = len(x_list)
    # in order to allow for iteration
    if marker_list == None: marker_list = [None,] * len(data_list)
    if label_list == None:
        label_list = ['Data {}'.format(i) for i, v in enumerate(data_list)] 
    for i,data in enumerate(data_list):
        # Clip data (3d plots are not good at it and show data out of plot)
        data_clip = data.copy()
        if axislim:
            j=0
            ###TODO: > complains about comparing with nan arising from <.
            # still gets the job done but try a workaround
            for dataj in data_clip.T:
                dataj[dataj<axislim[j][0]]=np.nan
                dataj[dataj>axislim[j][1]]=np.nan
                j+=1
        ax3d.scatter(data_clip[:,0],data_clip[:,1],data_clip[:,2],
                     s=10, marker=marker_list[i], label=label_list[i])
    ax3d.set_xlabel(xlabel,fontsize=9)
    ax3d.set_ylabel(ylabel,fontsize=9)
    ax3d.set_zlabel(zlabel,fontsize=9)
    # Draw spheres (mgrid 20 step-size j complex value for inclusive last point)
    if spheres:    
        theta, phi = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
        for i,ci in enumerate(spheres['centers']):
            x0,y0,z0 = ci
            r = spheres['radii'][i]
            x = x0 + r*np.cos(theta)*np.sin(phi)
            y = y0 + r*np.sin(theta)*np.sin(phi)
            z = z0 + r*np.cos(phi)
            ax3d.plot_surface(x,y,z,color='#d490c6',alpha=0.6)
    if axislim:
        ax3d.set_xlim(axislim[0])
        ax3d.set_ylim(axislim[1])
        ax3d.set_zlim(axislim[2])
    ax3d.set_aspect('equal')
    ax3d.legend(bbox_to_anchor=(0.5,-0.25), loc='upper center',ncol=2)
    return ax3d

def plot_projections(fig,axarr,data_list=False,
                     marker_list=['o',], shiftv=12, shifth=12,
                     label_list=None, axlabel_list=None, spheres=None,
                     axislim=None, legend_pad=0.2):
    # Load style file
    plt.style.reload_library()
    if 'paper' in plt.style.available: plt.style.use('paper')
    else: plt.style.use('ggplot')
#    # Transform single array of x into list of arrays with equal values
#    if not type(data_list) == list: data_list = [data_list,]
#    # Transform None and False *kwargs into list with len = len(y_list)
#    # in order to allow for iteration
#    if label_list == None:
#        label_list = ['y{}'.format(i) for i, v in enumerate(data_list)]    
    for row in range(axarr.shape[0]):
        i = row
        for col in range(axarr.shape[1]):
            j = col+1
            # Do what's needed on lower-left half of array and leave iteration
            if row>col:
                axarr[row,col].spines['left'].set_visible(False)
                axarr[row,col].spines['bottom'].set_visible(False)
                axarr[row,col].xaxis.set_ticks_position('none')
                axarr[row,col].yaxis.set_ticks_position('none')
                # Must get tick labels to adjust pad. Hide using white color
                axarr[row,col].tick_params(axis='x', labelcolor='w')
                axarr[row,col].tick_params(axis='y', labelcolor='w')
                if row+1==axarr.shape[0]:
                    axarr[row,col].set_xlabel(axlabel_list[j],labelpad=3)
                if col==0:
                    axarr[row,col].set_ylabel(axlabel_list[i],labelpad=3)
                continue
            # Show top and right axes and ticks (my 'paper' style removes them)
            axarr[row,col].spines['top'].set_visible(True)
            axarr[row,col].spines['right'].set_visible(True)
            axarr[row,col].xaxis.set_ticks_position('both')
            axarr[row,col].yaxis.set_ticks_position('both')
            # Set the grid (with transparency alpha)
            axarr[row,col].yaxis.grid(True, which='major')
            axarr[row,col].xaxis.grid(True, which='major')
            # Axes labels and tick labels only for outer plots
            if row==0:
                axarr[row,col].tick_params(labeltop=True,pad=3)
            if row+1==axarr.shape[0]:
                axarr[row,col].set_xlabel(axlabel_list[j],labelpad=3)
                axarr[row,col].xaxis.set_label_position('bottom')
                axarr[row,col].tick_params(labelbottom=True,pad=3)
            if col==0:
                axarr[row,col].set_ylabel(axlabel_list[i],labelpad=3)
                axarr[row,col].yaxis.set_label_position('left')
                axarr[row,col].tick_params(labelleft=True,pad=3)
            if col+1==axarr.shape[1]:
                axarr[row,col].tick_params(labelright=True,pad=3)
            # Minor tick location, dividing major in 2
            axarr[row,col].xaxis.set_minor_locator(AutoMinorLocator(n=2))
            axarr[row,col].yaxis.set_minor_locator(AutoMinorLocator(n=2))
            # Scatter plot
            if data_list:
                for idata,data in enumerate(data_list):
                    axarr[row,col].scatter(
                        data[:,i],data[:,j], marker=marker_list[idata],
                        label=label_list[idata], s=4
                        )
            # Plot spheres
            for ic,ci in enumerate(spheres['centers']):
                ri = spheres['radii'][ic]
                cplt = plt.Circle(ci[[i,j]],ri,color='#d490c6',alpha=0.25,ec='none')
                axarr[row,col].add_artist(cplt)
            # Legend (only above last column lower plot)
            if row+1==axarr.shape[0] and col+1==axarr.shape[1]:
                sphere_patch = mlines.Line2D([],[], linewidth=0, marker='o',
                    mfc='#d490c6', ms=10, alpha=0.5)
                handles, labels = axarr[row,col].get_legend_handles_labels()
                handles.append(sphere_patch)
                labels.append(r'$FPS$')
                shift_legend = (0.5+0.1)*(axarr.shape[1]-1)
                axarr[row,col].legend(handles,labels,
                     bbox_to_anchor=(0.5-shift_legend,-legend_pad),
                     loc='upper center',ncol=2)
            # Axis limits and equal size
            if not axislim:
                axislim = [(-0.5,1.5),]*(axarr.shape[0]+1)
            axarr[row,col].set_xlim(axislim[col])
            axarr[row,col].set_ylim(axislim[row])
            axarr[row,col].set_aspect('equal')
    
    # vertical line
    offset1 = mtransforms.ScaledTranslation(
        -shiftv/72., 0, fig.dpi_scale_trans)
    trans1 = mtransforms.blended_transform_factory(
        axarr[0,0].transAxes+offset1, fig.transFigure)
    l1 = mlines.Line2D([0,0], [0, 0.93], transform=trans1,
        figure=fig, color="k", linewidth=1.2, zorder=0)
    # horizontal line
    offset2 = mtransforms.ScaledTranslation(
        0,-shifth/72., fig.dpi_scale_trans)
    trans2 = mtransforms.blended_transform_factory(
        fig.transFigure, axarr[-1,0].transAxes+offset2)
    l2 = mlines.Line2D([0, 0.93], [0,0], transform=trans2,
        figure=fig, color="k", linewidth=1.2, zorder=0)
    # add lines to canvas
    fig.lines.extend([l1, l2])
    return fig, axarr

def plot_contour(ax,X,Y,Z, levels=6,values=None, norm=None,
                 xlabel=r'$x$',ylabel=r'$y$',title=None, colormap='winter',
                 fmt='%1.0f',inline_spacing=10):
    # Load style file
    plt.style.reload_library()
    if 'paper' in plt.style.available: plt.style.use('paper')
    else: plt.style.use('ggplot')
    # Plot
    #ax = plt.subplot2grid((4,1), (0, 0), rowspan=3)
    if values:
        CS = ax.contour(X,Y,Z,values,cmap=plt.get_cmap(colormap),norm=norm)
    else:
        CS = ax.contour(X,Y,Z,levels,cmap=plt.get_cmap(colormap))
    # Labels
    plt.clabel(CS, inline=True, inline_spacing=inline_spacing, fontsize=8, 
               fmt=fmt, colors='k')
    # Title
    if title: ax.set_title(title, y=1.02)
    # Axes labels from the names given in xlabel and ylabel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)   
    # Minor tick location, dividing major in 2
    ax.autoscale()
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))    
    # Set the grid with transparency alpha
    ax.yaxis.grid(True, which='major')
    ax.xaxis.grid(True, which='major')    
    return ax


# --- TEST RUN ---
if __name__ == '__main__':
    x = np.arange(0.0, 2.0, 0.1)
    y1 = np.sin(x); y2 = y1; y3 = np.cos(x); y4 = y3
    y5 = np.array([1,-1]*10)
    y_list = [y1,y2,y3,y4,y5]
    xlabel = 'x Axis '  r'$\left[ \/ \frac{\/m^2}{s} \/ \right]$'
    ylabel = 'y Axis ' r'$\left[ \/ kg\/m / s^2 \right]$'
    labels = ['y1','y2','y3','y4','y5']
    lines = ['none','-','none','-','steps']
    markers = ['^',None,'o',None,None]

    plt.figure(1)
    plot_lines(x, y_list, xlabel=xlabel, ylabel=ylabel,
            linestyle_list=lines, marker_list=markers,
            label_list=labels)

    plt.figure(2)
    ax,fig = plot_lines(x, [y1,], xlabel=xlabel, ylabel=ylabel, label_list=['y1'])
    
    #fig.savefig('testfig.pdf')
    
#    from matplotlib.ticker import LinearLocator, FormatStrFormatter
#    from mpl_toolkits.mplot3d import Axes3D
#    import matplotlib.pyplot as plt
#    import numpy as np
#    fig = plt.figure(3)
#    ax = fig.gca(projection='3d')
#    X = np.arange(-5, 5, 0.25)
#    Y = np.arange(-5, 5, 0.25)
#    X, Y = np.meshgrid(X, Y)
#    R = np.sqrt(X**2 + Y**2)
#    Z = np.sin(R)
#    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False)
#    ax.set_zlim(-1.01, 1.01)
#    
#    ax.zaxis.set_major_locator(LinearLocator(10))
#    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#    
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    
#    plt.show()