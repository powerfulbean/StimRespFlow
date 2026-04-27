import mne
import scipy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from statsmodels.stats.multitest import fdrcorrection 

def plot_biosemi128(r, title, chan_idx, folder = None, units = 'r', res = 1024, **kwargs):
    r = np.array(r)
    # print('TRFResult plot - r: ', r)
    assert len(r) == 128
    kwargs['sensors'] = True if 'sensors' not in kwargs else kwargs['sensors']
    fig = plot_data(
        r, title = title, chan_idx = chan_idx,  res = res, units = units, **kwargs
    )
    if 'ax' not in kwargs:
        fig.suptitle(title)
        if folder is not None:
            new_title = title.replace(" ","_").replace("\n","_")
            fig.savefig(f'{folder}/{new_title}.png', dpi = 300)
            plt.close(fig)

def plot_data(
    data,
    fs = 64,
    times = None,
    title = '',
    chan_idx = None,
    time_intvl = None,
    units = 'a.u.',
    montage = None,
    mode = 'joint',
    tmin = 0,
    
    **kwargs
):

    data = np.array(data)
    ifAx = False
    if montage is None:
        montage = mne.channels.make_standard_montage('biosemi128')
        chnames_map = dict(
            C17 = 'Fpz',
            # C21 = 'Fz',
            # A1 = 'Cz',
            D23 = 'T7',
            B26 = 'T8',
            # A19 = 'Pz',
            A23 = 'Oz'
        )

        for k,v in chnames_map.items():
            montage.ch_names[montage.ch_names.index(k)] = v

    chNames = montage.ch_names
    
    # print(chNames)
    try:
        info = mne.create_info(chNames, fs,ch_types = 'eeg', montage = montage)
    except:
        info = mne.create_info(chNames, fs,ch_types = 'eeg')
        info.set_montage(montage = montage)
    
    kwargs['sensors'] = False if 'sensors' not in kwargs else kwargs['sensors']
    kwargs['res'] = 256 if 'res' not in kwargs else kwargs['res']
    kwargs['outlines'] ='head' if 'outlines' not in kwargs else  kwargs['outlines']
    show_names = kwargs.get('show_names', False)
    if 'show_names' in kwargs:
        del kwargs['show_names']
    if show_names:
        names = montage.ch_names
    else:
        names = None
    ts_args = kwargs.get('ts_args', None)
    
    if time_intvl is not None:
        time_intvl = np.array(time_intvl)
        if time_intvl.ndim == 1:
            time_intvl = time_intvl[None,...]
        # print(time_intvl)
        #calculate the intersted time point and average time window
        average_window = time_intvl[:,1] - time_intvl[:,0]
        timepoint = time_intvl.mean(1)
    else:
        average_window = None
    
    chanMask = None
    if chan_idx is not None:
        chanMask = np.zeros(data.shape,dtype = bool)
        for i in chan_idx:
            chanMask[i] = True
    
    kwargs['cmap'] = plt.get_cmap("bwr") if 'cmap' not in kwargs else kwargs['cmap']
    kwargs['show'] = False if 'show' not in kwargs else kwargs['show']
    
    maskParam = dict(
        marker='o', 
        markerfacecolor='w', 
        markeredgecolor='k',
        linewidth=0, 
        markersize=8
    )
    
    maskParam2_default = dict(
        marker='o', 
        markerfacecolor='w', 
        markeredgecolor='k',
        linewidth=0, 
        markersize=4
    )
    maskParam2 = kwargs.get('maskParam', maskParam2_default)
    if 'maskParam' in kwargs:
        del kwargs['maskParam']
    
    if data.ndim == 2:

        default_ts_args={
            "units": units, 
            "scalings": dict(eeg=1),
            "highlight": time_intvl,
        }

        if ts_args is not None:
            default_ts_args.update(ts_args)


        mneW = mne.EvokedArray(data,info, tmin = tmin)
        if montage is not None:
            mneW.set_montage(montage)
        
        if times is None:
            if time_intvl is None:
                if mode == 'joint':
                    times = 'peaks'
                else:
                    times = 'auto'
            else:
                times = timepoint
        
        if mode == 'joint':
            default_topomap_args = dict(
                scalings = 1,
                mask = chanMask,
                mask_params= maskParam,
                average = average_window
            )

            if 'topomap_args' in kwargs:
                default_topomap_args.update(kwargs['topomap_args'])
                del kwargs['topomap_args']

            print(default_ts_args, default_topomap_args)
            fig = mneW.plot_joint(
                times = times,
                topomap_args=default_topomap_args,
                ts_args = default_ts_args,
                show = kwargs.get('show', True),
                title = title
            )
        else:
            fig = mneW.plot_topomap(
                times = times,
                time_unit='s',
                scalings = 1,
                title = title,
                units = units,
                cbar_fmt='%3.3f',
                mask = chanMask,
                mask_params= maskParam2, 
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
        # print('contours')

        if 'vlim' in kwargs:
            if kwargs['vlim'] == 'sym':
                t_d = data.squeeze()
                t_max = abs(t_d).max()
                kwargs['vlim'] = (-t_max, t_max)


        im,cm = mne.viz.plot_topomap(
            data.squeeze(),
            info,
            axes = ax1,
            mask = chanMask, 
            names = names,
            mask_params= maskParam2,
            sphere = 'eeglab',
            contours = 2,
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

def wilcoxon_fdr(r1,r2 = None, alternative = 'two-sided', ths = 0.05, fdr = True):
    r1 = np.array(r1)
    r2 = None if r2 is None else np.array(r2)
    if r1.ndim == 2:
        if r2 is not None:
            assert r1.shape == r2.shape
        imprv_stat = []
        for i in range(r1.shape[1]):
            if r2 is None:
                r2_cur = None
            else:
                r2_cur = r2[:,i]
            stat,p = scipy.stats.wilcoxon(r1[:,i], r2_cur, alternative = alternative)
            imprv_stat.append([stat,p])
        imprv_stat = np.array(imprv_stat)
        if fdr:
            rejected,pvalue_corrected = fdrcorrection(imprv_stat[:,1])
        else:
            pvalue_corrected = imprv_stat[:,1]
        chanIdx = np.where(pvalue_corrected <= ths)
        return imprv_stat,pvalue_corrected,chanIdx
    elif r1.ndim == 1:
        stat,p = scipy.stats.wilcoxon(r1, r2, alternative = alternative)
        return stat, p, p <= ths

def wilcoxon_fdr_test(
        x1, name1, x2, name2 = None, 
        alternative = 'two-sided', fdr = True,
        func_plot = plot_biosemi128, folder = None,
        verbose = False,
        if_return_stat = False,
        ths = 0.05,
        **kwargs
    ):
    if np.isscalar(x2):
        assert x2 == 0
        assert name2 is None
        x1 = np.array(x1)
        x2 = None
        name2 = 'zero'
        diff = x1
    else:
        # print(other.data.keys())
        x1 = np.array(x1)
        x2 = np.array(x2)
        diff = x1 - x2
        
    stat, p,chanIdx = wilcoxon_fdr(x1, x2, alternative, fdr = fdr, ths = ths) #pvalue_corrected,chanIdx
    title = f'{name1} - {name2} ({alternative})'
    if verbose:
        print(name1, name2, p, chanIdx)
    if x1.ndim == 2:
        rToPlot = diff.mean(0)
        func_plot(rToPlot, title, chanIdx, folder, **kwargs)
    else:
        rToPlot = diff.mean()
        with open(f"{folder}/{title}.txt", 'w') as f:
            f.write(f"if_sig: {chanIdx}, stat: {stat}, p: {p}, data: {diff}, data-mean: {diff.mean()}")
    
    # if folder:
    #     siIO.saveObject((p,(x1,x2)), rf'{folder}/{title}.tuple')
    
    if if_return_stat:
        return stat, p,chanIdx, diff
    else:
        return p,chanIdx, rToPlot