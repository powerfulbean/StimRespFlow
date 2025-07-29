import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import mne

def plotTopoplot(
    data,
    fs = 64,
    times = None,
    title = '',
    chanIdx = None,
    units = 'a.u.',
    montage = None,
    mode = 'joint',
    **kwargs
):
    data = np.array(data)
    ifAx = False
    if montage is None:
        montage = mne.channels.make_standard_montage('biosemi128')
        chNames = montage.ch_names

    try:
        info = mne.create_info(chNames, fs,ch_types = 'eeg', montage = montage)
    except:
        info = mne.create_info(chNames, fs,ch_types = 'eeg')
        info.set_montage(montage = montage)
    
    kwargs['sensors'] = False if 'sensors' not in kwargs else kwargs['sensors']
    kwargs['res'] = 256 if 'res' not in kwargs else kwargs['res']
    kwargs['cmap'] = plt.get_cmap("bwr") if 'cmap' not in kwargs else kwargs['cmap']
    
    chanMask = None
    if chanIdx is not None:
        chanMask = np.zeros(data.shape,dtype = bool)
        for i in chanIdx:
            chanMask[i] = True
    maskParam = dict(
        marker='o', 
        markerfacecolor='w', 
        markeredgecolor='k',
        linewidth=0, 
        markersize=8
    )
    
    if data.ndim == 2:
        mneW = mne.EvokedArray(data,info)
        if montage is not None:
            mneW.set_montage(montage)
        
        if times is None:
            if mode == 'joint':
                times = 'peaks'
            else:
                times = 'auto'
        
        if mode == 'joint':
            fig = mneW.plot_joint(
                 topomap_args={"scalings": 1}, 
                 ts_args={"units": units, "scalings": dict(eeg=1)}
            )
        else:
            fig = mneW.plot_topomap(
                times = times,
                outlines='head',
                time_unit='s',
                scalings = 1,
                title = title,
                units = units,
                cbar_fmt='%3.3f',
                mask = chanMask,
                mask_params= maskParam, 
                colorbar=False,
                # names = None,
                **kwargs
            )

    elif data.ndim == 1:
        if 'ax' in kwargs:
            ax1 = kwargs['ax']
            del kwargs['ax']
            ifAx = True
        else:
            fig = plt.figure(tight_layout=True)
            gridspec_kw={'width_ratios': [19, 1]}
            gs = gridspec.GridSpec(4, 2,**gridspec_kw)
            ax1 = fig.add_subplot(gs[:, 0])
            ax2 = fig.add_subplot(gs[1:3, 1])

        maskParam2 = dict(
            marker='o', 
            markerfacecolor='w', 
            markeredgecolor='k',
            linewidth=0, 
            markersize=4
        )

        im,cm = mne.viz.plot_topomap(
            data.squeeze(),
            info,
            outlines='head',
            axes = ax1,
            mask = chanMask, 
            names = None,
            mask_params= maskParam2,
            **kwargs
        )
        # cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        if not ifAx:
            clb = fig.colorbar(im, cax=ax2)
            clb.ax.set_title(units,fontsize=10) # title on top of colorbar
            fig.suptitle(title)
    else:
        raise NotImplementedError
    
    if not ifAx:
        return fig
    else:
        return im