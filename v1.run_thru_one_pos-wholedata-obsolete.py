#!/usr/bin/env python
# coding: utf-8

import importlib
import time
import pandas as pd
import numpy as np
import datetime
import shutil

from MERFISH_Objects.execute_class import *

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%m-%d %H:%M:%S', 
                    level=logging.INFO,
                    )

pbar = tqdm(0)
def update(*a):
    """take whatever results from myfunc and don't care
    """
    pbar.update()

def savefig(fig, path):
    """
    """
    fig.savefig(path, bbox_inches='tight', dpi=300)
    return 

def savefig_autodate(fig, path):
    """
    """
    today = datetime.today().date()
    suffix = path[-3:]
    assert suffix in ['pdf', 'png', 'jpg']
    path = path.replace(f'.{suffix}', f'_{today}.{suffix}')
    savefig(fig, path)
    print(f"saved the figure to: {path}")
    return 

def plot_image_with_zoom(img, x_window, y_window, vmin_p=0, vmax_p=100, title='', output=None):
    """
    """
    vmin,vmax = np.percentile(img.ravel(),[vmin_p, vmax_p])

    fig, axs = plt.subplots(1, 2, figsize=(10*2,10*1))
    for ax in axs:
        ax.set_title(title)
        g = ax.imshow(img,vmin=vmin,vmax=vmax,cmap='jet')

    ax = axs[0]
    ax.hlines(y_window, xmin=x_window[0], xmax=x_window[1], color='white')
    ax.vlines(x_window, ymin=y_window[0], ymax=y_window[1], color='white')

    ax = axs[1]
    ax.set_xlim(x_window)
    ax.set_ylim(y_window)
    fig.colorbar(g, shrink=0.5, ax=ax)
    if output is not None:
        savefig_autodate(fig, output)
    plt.show()

def pfunc_reg(hybe, metadata_path, dataset, posname, cword_config):
    self = Registration_Class(metadata_path,dataset,posname,hybe,cword_config,verbose=False)
    self.find_beads()

def pfunc_img(data, metadata_path, dataset, posname, cword_config):
    hybe = data['hybe']
    channel = data['channel']
    zindex = data['zindex']
    self = Image_Class(metadata_path,dataset,posname,hybe,channel,zindex,cword_config,verbose=False)
    self.main()

def pseudofunc_run_one_pos(posname, dataset, image_metadata_nuc, x_window, y_window, parameters, figdir, bitmap):
    """
    Caution: very sloppy - use many variables outside
    """
    pbar = tqdm(0) # dummy

    img_nuc = image_metadata_nuc.stkread(Position=posname, Channel='DeepBlue').max(2) # max across z?
    output = os.path.join(figdir, f'fig1-1_dapi_{dataset}_{posname}.pdf')
    logging.info(output)
    plot_image_with_zoom(img_nuc, x_window, y_window, vmin_p=25, vmax_p=95, title=posname+" nuclei", 
                        output=output,
                        )

    img_poly = image_metadata_nuc.stkread(Position=posname, Channel='FarRed').max(2) # max across z?
    output = os.path.join(figdir, f'fig1-2_polyT_{dataset}_{posname}.pdf')
    logging.info(output)
    plot_image_with_zoom(img_poly, x_window, y_window, vmin_p=25, vmax_p=95, title=posname+" polyT",
                        output=output,
                        )

    ### init takes sometime
    self = Dataset_Class(metadata_path, dataset, cword_config, verbose=True)
    self.main()

    # Registration 
    Input = [hybe for readout_probe,hybe,channel in bitmap if not hybe==parameters['ref_hybe']]

    logging.info("Registration")
    pfunc_reg(parameters['ref_hybe'], metadata_path, dataset, posname, cword_config)

    ncpu = min(len(Input), 6)
    pool = multiprocessing.Pool(ncpu)
    pbar.reset()
    pbar = tqdm(Input)
    for i in Input:
        pool.apply_async(pfunc_reg, args=(i, metadata_path, dataset, posname, cword_config), callback=update)
    pool.close()
    pool.join()
    
    ### Image
    logging.info("Image and spot proc")
    Input = []
    hybe = 'hybe2'
    channel = 'FarRed'
    self = Stack_Class(metadata_path,dataset,posname,hybe,channel,cword_config,verbose=False)
    self.check_projection()
    zindexes = self.zindexes
    for readout_probe,hybe,channel in bitmap:
        for zindex in zindexes:
            data = {'hybe':hybe,'channel':channel,'zindex':zindex}
            Input.append(data)

    ncpu = min(len(Input), 6)
    pool = multiprocessing.Pool(ncpu)
    pbar.reset()
    pbar = tqdm(Input)
    for i in Input:
        pool.apply_async(pfunc_img, args=(i, metadata_path, dataset, posname, cword_config), callback=update)
    pool.close()
    pool.join()

    # Get results and plot 
    nbits = len(bitmap)
    zindex = str(zindexes[0])
    rawimgs = {}
    imgs = {}
    spots_coords = {}

    for i, (readout,hybe,channel) in enumerate(bitmap):
        print(readout,hybe,channel)
        """ Processed Image Zoom"""
        self = Image_Class(metadata_path,dataset,posname,hybe,channel,zindex,cword_config,verbose=False)
        self.load_data() # sub_stk
        # stks[i] = self.sub_stk
        self.project() # max projection
        rawimgs[i] = self.img
        
        fish_img = self.fishdata.load_data('image',
                                            dataset=self.dataset,
                                            posname=self.posname,
                                            hybe=self.hybe,
                                            channel=self.channel,
                                            zindex=self.zindex)/self.parameters['gain']
        imgs[i] = fish_img
        self.img = fish_img # trying this and see; this is important
        
        self.parameters['spot_max_distance'] = 3#self.parameters['spot_parameters']['default']['spot_max_distance']
        self.parameters['spot_minmass'] = 5#self.parameters['spot_parameters']['default']['spot_minmass']
        self.parameters['spot_diameter'] = 5#self.parameters['spot_parameters']['default']['spot_diameter']
        self.parameters['spot_separation'] = 3#self.parameters['spot_parameters']['default']['spot_separation']
        self.call_spots()
        spots_coords[i] = np.vstack([self.spots.x, self.spots.y]).T

    # plots
    output = os.path.join(figdir, f'fig2-1_raw_{dataset}_{posname}.pdf')
    logging.info(output)
    fig, axs = plt.subplots(3,6,figsize=(4*6,4*3), sharex=True, sharey=True)
    for i, ax in zip(rawimgs.keys(), axs.flat):
        img = rawimgs[i]
        vmin = np.percentile(img.reshape(-1,), 50)
        vmax = np.percentile(img.reshape(-1,), 95)
        g = ax.imshow(img, cmap='jet', vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(f'bit{i}')
        fig.colorbar(g, ax=ax, shrink=0.5, location='bottom', pad=0.05, fraction=0.05, ticks=[vmin, vmax])
    fig.subplots_adjust(wspace=0)
    fig.suptitle(f"Raw images (Max projected) {posname}", y=0.92)
    savefig_autodate(fig, output)
    plt.show()

    output = os.path.join(figdir, f'fig2-2_proc_{dataset}_{posname}.pdf')
    logging.info(output)
    fig, axs = plt.subplots(3,6,figsize=(4*6,4*3), sharex=True, sharey=True)
    for i, ax in zip(imgs.keys(), axs.flat):
        img = imgs[i]
        vmin = np.percentile(img.reshape(-1,), 50)
        vmax = np.percentile(img.reshape(-1,), 95)
        g = ax.imshow(img, cmap='jet', vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(f'bit{i}')
        fig.colorbar(g, ax=ax, shrink=0.5, location='bottom', pad=0.05, fraction=0.05, ticks=[vmin, vmax])
    fig.subplots_adjust(wspace=0)
    fig.suptitle(f"FISHdata Images {posname}", y=0.92)
    savefig_autodate(fig, output)
    plt.show()

    output = os.path.join(figdir, f'fig2-3_spots_{dataset}_{posname}.pdf')
    logging.info(output)
    fig, axs = plt.subplots(3,6,figsize=(4*6,4*3), sharex=True, sharey=True)
    for i, ax in zip(spots_coords.keys(), axs.flat):
        spots = spots_coords[i]
        ax.scatter(spots[:,0], spots[:,1], s=2, edgecolor='none', rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(f'bit{i}')
    fig.subplots_adjust(wspace=0)
    fig.suptitle(f"Spots {posname}", y=0.92)
    savefig_autodate(fig, output)
    plt.show()

    # barcodes
    zindex = 0 # this is very important (otherwise z='0')
    self = Classify_Class(metadata_path,dataset,posname,zindex,cword_config,verbose=True)
    self.parameters['spot_max_distance'] = 20
    self.parameters['spot_minmass'] = 5
    self.parameters['spot_diameter'] = 5
    self.parameters['spot_separation'] = 5

    self.call_spots()
    self.pair_spots()
    self.build_barcodes()
    self.assign_codewords()
    self.collapse_spots()

    codebook = self.merfish_config.codebook
    transcripts = self.transcripts
    transcripts['gene'] = codebook.index[transcripts.cword_idx]
    transcripts_idx = transcripts['idx'].astype(int)
    cond = transcripts['gene'].str.contains("^blank")
    A = transcripts[cond] #.shape, 
    B = transcripts[~cond] #.shape
    obs_blank = len(A)/(len(B)+len(A))
    obsoverexp_blank = obs_blank/(17/187)
    measured_barcodes = self.measured_barcodes.numpy()
    updated_measured_barcodes = self.updatad_measured_barcodes.numpy()

    output = os.path.join(figdir, f'fig3-1_barcodes_{dataset}_{posname}.pdf')
    logging.info(output)
    fig, ax = plt.subplots(figsize=(10,5))
    val = measured_barcodes.sum(axis=0).astype(int)
    ax.plot(val, 'o', color='gray', label='measured; all')

    val = measured_barcodes[transcripts_idx].sum(axis=0).astype(int)
    ax.plot(val, 'o', color='k', label='measured; transcripts')

    val = updated_measured_barcodes.sum(axis=0).astype(int)
    ax.plot(val, 'o', label='updated; all')

    val = updated_measured_barcodes[transcripts_idx].sum(axis=0).astype(int)
    ax.plot(val, 'o', label='updated; transcripts')
    ax.set_title(f'{dataset}; {posname} \n Observed blank: {100*obs_blank:.1f}%; Obs/Exp: {obsoverexp_blank:.2f}')

    ax.set_xticks(np.arange(18))
    ax.set_xlabel('bit/hybe')
    ax.set_ylabel('number of positives')
    ax.legend(bbox_to_anchor=(1,1))
    savefig_autodate(fig, output)
    plt.show()

    output = os.path.join(figdir, f'fig3-2_barcodes_{dataset}_{posname}.pdf')
    logging.info(output)
    fig, axs = plt.subplots(1,2,figsize=(7*2,7), sharex=True, sharey=True)
    ax = axs[0]
    ax.set_title('Detected Transcripts')
    c=transcripts['signal-noise']
    vmin,vmax= np.percentile(c, [5,95])
    g = ax.scatter(transcripts.x,
                transcripts.y,
                c=c,
                s=1,
                vmin=vmin,
                vmax=vmax,
                cmap='jet', 
                rasterized=True,
                )
    fig.colorbar(g, ax=ax)
    ax = axs[1]
    ax.set_title('Detected Transcripts')
    ax.scatter(transcripts.x,
            transcripts.y,
            c='gray',
            s=1,
            label='all',
            rasterized=True,
            )
    ax.scatter(A.x,
            A.y,
            c='C1',
            s=2,
            rasterized=True,
            label='blanks',
            )
    ax.legend()
    savefig_autodate(fig, output)
    plt.show()


def main(figdir, metadata_path, cword_config, overwrite=True):
    """
    """
    config = importlib.import_module(cword_config)
    dataset = [i for i in metadata_path.split('/') if not i==''][-1]
    parameters = config.parameters
    fishdata_path = os.path.join(metadata_path, parameters['fishdata'])
    utilities_path = parameters['utilities_path']
    bitmap = config.bitmap

    for key in parameters:
        if key.endswith('path') or key.endswith('data'):
            print(key, parameters[key])
            
    nuclei = [i for i in os.listdir(metadata_path) if 'nucstain' in i][-1]
    print(nuclei)
    hybe1 = [i for i in os.listdir(metadata_path) if 'hybe1_' in i][-1]
    print(hybe1)

    image_metadata_nuc = Metadata(os.path.join(metadata_path, nuclei))
    print(image_metadata_nuc.posnames.shape)
    image_metadata = Metadata(os.path.join(metadata_path, hybe1))
    print(image_metadata.posnames.shape)

    posnames = image_metadata_nuc.posnames
    x_window = [500,750]
    y_window = [1000,1250]

    # need to clean up temporary folder
    for posname in posnames:
        if overwrite:
            try:
                shutil.rmtree(fishdata_path)
            except Exception as e:
                print(e)
            try:
                shutil.rmtree(utilities_path)
            except Exception as e:
                print(e)
        pseudofunc_run_one_pos(posname, dataset, image_metadata_nuc, x_window, y_window, parameters, figdir, bitmap)

        break ### !!!! this is to test


if __name__ == '__main__':
    ### input
    figdir = '/bigstore/GeneralStorage/fangming/projects/test_merfish/figures'
    cword_config = 'merfish_config_zebrafinch_18bits_bigruns'
    overwrite = True # True # False # True should be the default
    metadata_paths = [
        '/bigstore/Images2022/Gaby/Zebrafinch/C0_2022Jul25/',
        # '/bigstore/Images2022/Gaby/Zebrafinch/Zebra_B0_2022Jul11/',
        # '/bigstore/Images2022/Gaby/Zebrafinch/A3_2022Feb29/',
    ]
    for metadata_path in metadata_paths:
        main(figdir, metadata_path, cword_config, overwrite=overwrite)