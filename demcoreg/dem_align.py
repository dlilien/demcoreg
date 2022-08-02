#! /usr/bin/env python

#Todo
#Better outlier removal
#Check Nuth and Kaab bin median
#Implement check for empty diff

import sys
import os
import argparse
import subprocess

from osgeo import gdal
import numpy as np

from pygeotools.lib import iolib, malib, geolib, warplib, filtlib

from demcoreg import coreglib, dem_mask

#Turn off numpy multithreading
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

def get_mask(ds, mask_list, dem_fn=None):
    #This returns True (1) for areas to mask, False (0) for valid static surfaces
    static_mask = dem_mask.get_mask(ds, mask_list, dem_fn, writeout=False)
    #return ~(static_mask)
    return static_mask

def outlier_filter(diff, f=3, perc=None, max_dz=100):
    print("Removing outliers")
    print("Initial pixel count:", diff.count())
    print("Absolute dz filter: %0.2f" % max_dz)
    #Absolute dz filter
    diff = np.ma.masked_greater(diff, max_dz)

    if perc is not None:
        diff = filtlib.perc_fltr(diff, perc)
    else:
        #diff = filtlib.sigma_fltr(diff, f)
        diff = filtlib.mad_fltr(diff, f)

    print("Filtered pixel count:", diff.count())
    return diff

def get_filtered_slope(ds, slope_lim=(0.01, 40)):
    #Generate slope map
    print("Computing slope")
    slope = geolib.gdaldem_mem_ds(ds, processing='slope', returnma=True, computeEdges=False)
    #slope_stats = malib.print_stats(slope)
    print("Slope filter: %0.2f - %0.2f" % slope_lim)
    print("Initial count: %i" % slope.count()) 
    slope = filtlib.range_fltr(slope, slope_lim) 
    print("Filtered count: %i" % slope.count()) 
    return slope

def compute_offset(ref_dem_ds, src_dem_ds, src_dem_fn, mode='nuth', remove_outliers=True, max_offset=100, \
        max_dz=100, slope_lim=(0.1, 40), mask_list=['glaciers',]):
    #Make sure the input datasets have the same resolution/extent
    #Use projection of source DEM
    ref_dem_clip_ds, src_dem_clip_ds = warplib.memwarp_multi([ref_dem_ds, src_dem_ds], \
            res='max', extent='intersection', t_srs=src_dem_ds, r='cubic')

    #Compute size of NCC and SAD search window in pixels
    res = float(geolib.get_res(ref_dem_clip_ds, square=True)[0])
    max_offset_px = (max_offset/res) + 1
    pad = (int(max_offset_px), int(max_offset_px))

    #This will be updated geotransform for src_dem
    src_dem_gt = np.array(src_dem_clip_ds.GetGeoTransform())

    #Load the arrays
    ref_dem = iolib.ds_getma(ref_dem_clip_ds, 1)
    src_dem = iolib.ds_getma(src_dem_clip_ds, 1)

    print("Elevation difference stats for uncorrected input DEMs (src - ref)")
    diff = src_dem - ref_dem

    static_mask = get_mask(src_dem_clip_ds, mask_list, src_dem_fn)
    diff = np.ma.array(diff, mask=static_mask)

    if diff.count() == 0:
        sys.exit("No overlapping, unmasked pixels shared between input DEMs")

    if remove_outliers:
        diff = outlier_filter(diff, f=3, max_dz=max_dz)

    #Want to use higher quality DEM, should determine automatically from original res/count
    #slope = get_filtered_slope(ref_dem_clip_ds, slope_lim=slope_lim)
    slope = get_filtered_slope(src_dem_clip_ds, slope_lim=slope_lim)

    print("Computing aspect")
    #aspect = geolib.gdaldem_mem_ds(ref_dem_clip_ds, processing='aspect', returnma=True, computeEdges=False)
    aspect = geolib.gdaldem_mem_ds(src_dem_clip_ds, processing='aspect', returnma=True, computeEdges=False)

    ref_dem_clip_ds = None
    src_dem_clip_ds = None

    #Apply slope filter to diff
    #Note that we combine masks from diff and slope in coreglib
    diff = np.ma.array(diff, mask=np.ma.getmaskarray(slope))

    #Get final mask after filtering
    static_mask = np.ma.getmaskarray(diff)

    #Compute stats for new masked difference map
    print("Filtered difference map")
    diff_stats = malib.print_stats(diff)
    dz = diff_stats[5]

    print("Computing sub-pixel offset between DEMs using mode: %s" % mode)

    #By default, don't create output figure
    fig = None

    #Default horizntal shift is (0,0)
    dx = 0
    dy = 0

    #Sum of absolute differences
    if mode == "sad":
        ref_dem = np.ma.array(ref_dem, mask=static_mask)
        src_dem = np.ma.array(src_dem, mask=static_mask)
        m, int_offset, sp_offset = coreglib.compute_offset_sad(ref_dem, src_dem, pad=pad)
        #Geotransform has negative y resolution, so don't need negative sign
        #np array is positive down
        #GDAL coordinates are positive up
        dx = sp_offset[1]*src_dem_gt[1]
        dy = sp_offset[0]*src_dem_gt[5]
    #Normalized cross-correlation of clipped, overlapping areas
    elif mode == "ncc":
        ref_dem = np.ma.array(ref_dem, mask=static_mask)
        src_dem = np.ma.array(src_dem, mask=static_mask)
        m, int_offset, sp_offset, fig = coreglib.compute_offset_ncc(ref_dem, src_dem, \
                pad=pad, prefilter=False)
        dx = sp_offset[1]*src_dem_gt[1]
        dy = sp_offset[0]*src_dem_gt[5]
    #Nuth and Kaab (2011)
    elif mode == "nuth":
        #Compute relationship between elevation difference, slope and aspect
        fit_param, fig = coreglib.compute_offset_nuth(diff, slope, aspect)
        if fit_param is None:
            print("Failed to calculate horizontal shift")
        else:
            #fit_param[0] is magnitude of shift vector
            #fit_param[1] is direction of shift vector
            #fit_param[2] is mean bias divided by tangent of mean slope
            dx = fit_param[0]*np.sin(np.deg2rad(fit_param[1]))
            dy = fit_param[0]*np.cos(np.deg2rad(fit_param[1]))
            med_slope = malib.fast_median(slope)
            nuth_dz = fit_param[2]*np.tan(np.deg2rad(med_slope))
            print('Median dz: %0.2f\nNuth dz: %0.2f' % (dz, nuth_dz))
            #dz = nuth_dz

    return -dx, -dy, -dz, static_mask, fig

def getparser():
    parser = argparse.ArgumentParser(description="Perform DEM co-registration using multiple algorithms", \
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ref_fn', type=str, help='Reference DEM filename')
    parser.add_argument('src_fn', type=str, help='Source DEM filename to be shifted')
    parser.add_argument('-mode', type=str, default='nuth', choices=['ncc', 'sad', 'nuth', 'none'], \
            help='Type of co-registration to use')
    parser.add_argument('-mask_list', nargs='+', type=str, default=[], choices=dem_mask.mask_choices, \
            help='Define masks to use to limit reference surfaces for co-registration')
    parser.add_argument('-tiltcorr', action='store_true', \
            help='After preliminary translation, fit polynomial to residual elevation offsets and remove')
    parser.add_argument('-polyorder', type=int, default=1, \
            help='Specify order of polynomial fit') 
    parser.add_argument('-tol', type=float, default=0.02, \
            help='When iterative translation magnitude is below this tolerance (meters), break and write out corrected DEM')
    parser.add_argument('-max_offset', type=float, default=100, \
            help='Maximum expected horizontal offset in meters, used to set search range for ncc and sad modes')
    parser.add_argument('-max_dz', type=float, default=100, \
            help='Maximum expected vertical offset in meters, used to filter outliers')
    res_choices = ['min', 'max', 'mean', 'common_scale_factor']
    parser.add_argument('-res', type=str, default='mean', choices=res_choices, \
            help='Warp intputs to this resolution') 
    parser.add_argument('-slope_lim', type=float, nargs=2, default=(0.1, 40), \
            help='Minimum and maximum surface slope limits to consider')
    parser.add_argument('-max_iter', type=int, default=30, \
            help='Maximum number of iterations, if tol is not reached')
    parser.add_argument('-outdir', default=None, help='Output directory')
    return parser

def main(argv=None):
    parser = getparser()
    args = parser.parse_args()

    #Should check that files exist
    ref_dem_fn = args.ref_fn
    src_dem_fn = args.src_fn

    mode = args.mode
    mask_list = args.mask_list
    max_offset = args.max_offset
    max_dz = args.max_dz
    slope_lim = tuple(args.slope_lim)
    tiltcorr = args.tiltcorr
    polyorder = args.polyorder
    res = args.res

    #Maximum number of iterations
    max_iter = args.max_iter

    #These are tolerances (in meters) to stop iteration
    tol = args.tol
    min_dx = tol
    min_dy = tol
    min_dz = tol

    outdir = args.outdir
    if outdir is None:
        outdir = os.path.splitext(src_dem_fn)[0] + '_dem_align'

    if tiltcorr:
        outdir += '_tiltcorr'
        tiltcorr_done = False
        #Relax tolerance for initial round of co-registration
        #tiltcorr_tol = 0.1
        #if tol < tiltcorr_tol:
        #    tol = tiltcorr_tol

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outprefix = '%s_%s' % (os.path.splitext(os.path.split(src_dem_fn)[-1])[0], \
            os.path.splitext(os.path.split(ref_dem_fn)[-1])[0])
    outprefix = os.path.join(outdir, outprefix)

    print("\nReference: %s" % ref_dem_fn)
    print("Source: %s" % src_dem_fn)
    print("Mode: %s" % mode)
    print("Output: %s\n" % outprefix)

    src_dem_ds = gdal.Open(src_dem_fn)
    ref_dem_ds = gdal.Open(ref_dem_fn)

    #Get local cartesian coordinate system
    #local_srs = geolib.localtmerc_ds(src_dem_ds)
    #Use original source dataset coordinate system
    #Potentially issues with distortion and xyz/tiltcorr offsets for DEM with large extent
    local_srs = geolib.get_ds_srs(src_dem_ds)
    #local_srs = geolib.get_ds_srs(ref_dem_ds)

    #Resample to common grid
    ref_dem_res = float(geolib.get_res(ref_dem_ds, t_srs=local_srs, square=True)[0])
    #Create a copy to be updated in place
    src_dem_ds_align = iolib.mem_drv.CreateCopy('', src_dem_ds, 0)
    src_dem_res = float(geolib.get_res(src_dem_ds, t_srs=local_srs, square=True)[0])
    src_dem_ds = None
    #Resample to user-specified resolution
    ref_dem_ds, src_dem_ds_align = warplib.memwarp_multi([ref_dem_ds, src_dem_ds_align], \
            extent='intersection', res=args.res, t_srs=local_srs, r='cubic')

    res = float(geolib.get_res(src_dem_ds_align, square=True)[0])
    print("\nReference DEM res: %0.2f" % ref_dem_res)
    print("Source DEM res: %0.2f" % src_dem_res)
    print("Resolution for coreg: %s (%0.2f m)\n" % (args.res, res))

    #Iteration number
    n = 1
    #Cumulative offsets
    dx_total = 0
    dy_total = 0
    dz_total = 0

    #Now iteratively update geotransform and vertical shift
    while True:
        print("*** Iteration %i ***" % n)
        dx, dy, dz, static_mask, fig = compute_offset(ref_dem_ds, src_dem_ds_align, src_dem_fn, mode, max_offset, \
                mask_list=mask_list, max_dz=max_dz, slope_lim=slope_lim)
        xyz_shift_str_iter = "dx=%+0.2fm, dy=%+0.2fm, dz=%+0.2fm" % (dx, dy, dz)
        print("Incremental offset: %s" % xyz_shift_str_iter)

        dx_total += dx
        dy_total += dy
        dz_total += dz

        xyz_shift_str_cum = "dx=%+0.2fm, dy=%+0.2fm, dz=%+0.2fm" % (dx_total, dy_total, dz_total)
        print("Cumulative offset: %s" % xyz_shift_str_cum)
        #String to append to output filenames
        xyz_shift_str_cum_fn = '_%s_x%+0.2f_y%+0.2f_z%+0.2f' % (mode, dx_total, dy_total, dz_total)

        #Apply the horizontal shift to the original dataset
        src_dem_ds_align = coreglib.apply_xy_shift(src_dem_ds_align, dx, dy, createcopy=False)
        #Should 
        src_dem_ds_align = coreglib.apply_z_shift(src_dem_ds_align, dz, createcopy=False)

        n += 1
        print("\n")
        #If magnitude of shift in all directions is less than tol
        #if n > max_iter or (abs(dx) <= min_dx and abs(dy) <= min_dy and abs(dz) <= min_dz):
        #If magnitude of shift is less than tol
        dm = np.sqrt(dx**2 + dy**2 + dz**2)
        dm_total = np.sqrt(dx_total**2 + dy_total**2 + dz_total**2)

        if dm_total > max_offset:
            sys.exit("Total offset exceeded specified max_offset (%0.2f m). Consider increasing -max_offset argument" % max_offset)

        #Stop iteration
        if n > max_iter or dm < tol:
            if True:
                ref_dem_clip_ds_align, src_dem_clip_ds_align = warplib.memwarp_multi([ref_dem_ds, src_dem_ds_align], \
                        res=res, extent='intersection', t_srs=local_srs, r='cubic')
                ref_dem_align = iolib.ds_getma(ref_dem_clip_ds_align, 1)
                src_dem_align = iolib.ds_getma(src_dem_clip_ds_align, 1)
                ref_dem_clip_ds_align = None

                diff_align = src_dem_align - ref_dem_align
                src_dem_align = None
                ref_dem_align = None

                #Get updated, final mask
                static_mask_final = get_mask(src_dem_clip_ds_align, mask_list, src_dem_fn)
                static_mask_final = np.logical_or(np.ma.getmaskarray(diff_align), static_mask_final)
                
                #Final stats, before outlier removal
                diff_align_compressed = diff_align[~static_mask_final]
                diff_align_stats = malib.get_stats_dict(diff_align_compressed, full=True)

                #Prepare filtered version for tiltcorr fit
                diff_align_filt = np.ma.array(diff_align, mask=static_mask_final)
                diff_align_filt = outlier_filter(diff_align_filt, f=3, max_dz=max_dz)
                #diff_align_filt = outlier_filter(diff_align_filt, perc=(12.5, 87.5), max_dz=max_dz)
                slope = get_filtered_slope(src_dem_clip_ds_align)
                diff_align_filt = np.ma.array(diff_align_filt, mask=np.ma.getmaskarray(slope))
                diff_align_filt_stats = malib.get_stats_dict(diff_align_filt, full=True)

            #Fit 2D polynomial to residuals and remove
            #To do: add support for along-track and cross-track artifacts
            if tiltcorr and not tiltcorr_done:
                print("\n************")
                print("Calculating 'tiltcorr' 2D polynomial fit to residuals with order %i" % polyorder)
                print("************\n")
                gt = src_dem_clip_ds_align.GetGeoTransform()

                #Need to apply the mask here, so we're only fitting over static surfaces
                #Note that the origmask=False will compute vals for all x and y indices, which is what we want
                vals, resid, coeff = geolib.ma_fitpoly(diff_align_filt, order=polyorder, gt=gt, perc=(0,100), origmask=False)
                #vals, resid, coeff = geolib.ma_fitplane(diff_align_filt, gt, perc=(12.5, 87.5), origmask=False)

                #Should write out coeff or grid with correction 

                vals_stats = malib.get_stats_dict(vals)

                #Want to have max_tilt check here
                #max_tilt = 4.0 #m
                #Should do percentage
                #vals.ptp() > max_tilt

                #Note: dimensions of ds and vals will be different as vals are computed for clipped intersection
                #Need to recompute planar offset for full src_dem_ds_align extent and apply
                xgrid, ygrid = geolib.get_xy_grids(src_dem_ds_align)
                valgrid = geolib.polyval2d(xgrid, ygrid, coeff) 
                #For results of ma_fitplane
                #valgrid = coeff[0]*xgrid + coeff[1]*ygrid + coeff[2]
                src_dem_ds_align = coreglib.apply_z_shift(src_dem_ds_align, -valgrid, createcopy=False)

                print("Applying tilt correction to difference map")
                diff_align -= vals

                #Should iterate until tilts are below some threshold
                #For now, only do one tiltcorr
                tiltcorr_done=True
                #Now use original tolerance, and number of iterations 
                tol = args.tol
                max_iter = n + args.max_iter
            else:
                break

    if True:
        #Write out aligned difference map for clipped extent with vertial offset removed
        align_diff_fn = outprefix + '%s_align_diff.tif' % xyz_shift_str_cum_fn
        print("Writing out aligned difference map with median vertical offset removed")
        iolib.writeGTiff(diff_align, align_diff_fn, src_dem_clip_ds_align)

    #Extract final center coordinates for intersection
    center_coord_ll = geolib.get_center(src_dem_clip_ds_align, t_srs=geolib.wgs_srs)
    center_coord_xy = geolib.get_center(src_dem_clip_ds_align)

    #Write out final aligned src_dem 
    align_fn = outprefix + '%s_align.tif' % xyz_shift_str_cum_fn
    print("Writing out shifted src_dem with median vertical offset removed: %s" % align_fn)
    #Open original uncorrected dataset at native resolution
    src_dem_ds = gdal.Open(src_dem_fn)


    src_dem_ds_align = iolib.mem_drv.CreateCopy('', src_dem_ds, 0)
    #Apply final horizontal and vertial shift to the original dataset
    #Note: potentially issues if we used a different projection during coregistration!
    src_dem_ds_align = coreglib.apply_xy_shift(src_dem_ds_align, dx_total, dy_total, createcopy=False)
    src_dem_ds_align = coreglib.apply_z_shift(src_dem_ds_align, dz_total, createcopy=False)
    if tiltcorr:
        xgrid, ygrid = geolib.get_xy_grids(src_dem_ds_align)
        valgrid = geolib.polyval2d(xgrid, ygrid, coeff) 
        #For results of ma_fitplane
        #valgrid = coeff[0]*xgrid + coeff[1]*ygrid + coeff[2]
        src_dem_ds_align = coreglib.apply_z_shift(src_dem_ds_align, -valgrid, createcopy=False)
    #Might be cleaner way to write out MEM ds directly to disk
    src_dem_full_align = iolib.ds_getma(src_dem_ds_align)
    iolib.writeGTiff(src_dem_full_align, align_fn, src_dem_ds_align)
    src_dem_clip_ds_align = None

    ref_dem_full_ds_align, src_dem_full_ds_align = warplib.memwarp_multi([ref_dem_ds, src_dem_ds_align], \
                        res='last', extent='last', t_srs=local_srs, r='cubic')
    ref_dem_full_align = iolib.ds_getma(ref_dem_full_ds_align, 1)

    if True:
        #Output final aligned src_dem, masked so only best pixels are preserved
        #Useful if creating a new reference product
        #Can also use apply_mask.py 
        diff_align = src_dem_full_align - ref_dem_full_align
        diff_align_filt = np.ma.array(diff_align)
        diff_align_filt = outlier_filter(diff_align_filt, f=3, max_dz=max_dz)
        #diff_align_filt = outlier_filter(diff_align_filt, perc=(12.5, 87.5), max_dz=max_dz)
        slope = get_filtered_slope(src_dem_ds_align)
        diff_align_filt = np.ma.array(diff_align_filt, mask=np.ma.getmaskarray(slope))

        print("Applying filter to shifted src_dem")
        align_fn_masked = outprefix + '%s_align_filt.tif' % xyz_shift_str_cum_fn
        iolib.writeGTiff(np.ma.array(src_dem_full_align, mask=np.ma.getmaskarray(diff_align_filt)), \
                align_fn_masked, src_dem_ds_align)

    src_dem_full_align = None
    src_dem_ds_align = None


if __name__ == "__main__":
    main()
