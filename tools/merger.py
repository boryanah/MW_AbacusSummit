import asdf
import numpy as np
import scipy.stats as scist
import matplotlib.pyplot as plt
from astropy.table import Table
from numba import jit

def extract_superslab(fn):
    # looks like "associations_z0.100.0.asdf"
    return int(fn.split('.')[-2])

def extract_superslab_minified(fn):
    # looks like "associations_z0.100.0.asdf.minified"
    return int(fn.split('.')[-3])
    
def extract_redshift(fn):
    # looks like "associations_z0.100.0.asdf.minified" or "associations_z0.100.0.asdf"
    redshift = float('.'.join(fn.split("z")[-1].split('.')[:2]))
    return redshift

def get_zs_from_headers(snap_names):
    '''
    Read redshifts from merger tree files
    '''
    
    zs = np.zeros(len(snap_names))
    for i in range(len(snap_names)):
        snap_name = snap_names[i]
        with asdf.open(snap_name) as f:
            zs[i] = np.float(f["header"]["Redshift"])
    return zs

def get_one_header(merger_dir):
    '''
    Get an example header by looking at one association
    file in a merger directory
    '''

    # choose one of the merger tree files
    fn = list(merger_dir.glob('associations*.asdf'))[0]
    with asdf.open(fn) as af:
        header = af['header']
    return header

def unpack_inds(halo_ids):
    '''
    Unpack indices in Sownak's format of Nslice*1e12 
    + superSlabNum*1e9 + halo_position_superSlab
    '''
    
    # obtain slab number and index within slab
    id_factor = int(1e12)
    slab_factor = int(1e9)
    index = (halo_ids % slab_factor).astype(int)
    slab_number = ((halo_ids % id_factor - index) // slab_factor).astype(int)
    return slab_number, index

def pack_inds(halo_ids, slab_ids):
    '''
    Pack indices in Sownak's format of Nslice*1e12 
    + superSlabNum*1e9 + halo_position_superSlab
    '''
    # just as a place holder
    slice_ids = 0
    halo_ids = slab_ids*1e9 + halo_ids
    return halo_ids

def reorder_by_slab(fns,minified):
    '''
    Reorder filenames in terms of their slab number
    '''
    if minified:
        return sorted(fns, key=extract_superslab_minified)
    else:
        return sorted(fns, key=extract_superslab)

@jit(nopython = True)
def count_progenitors(nums, starts, main_progs, progs, masses_prev,
                    offsets, slabs_prev, m_low, m_high):
    # constants used for unpacking
    id_factor = int(1e12)
    slab_factor = int(1e9)

    # number of objects of interest
    N_this = len(nums)
    N_merger = np.zeros(len(nums), dtype=np.int64)
    # loop around halos that were marked belonging to this redshift catalog
    for j in range(N_this):
        # select all progenitors
        start = starts[j]
        num = nums[j]
        mp = main_progs[j]
        prog_inds = progs[start : start + num]

        # remove progenitors with no info
        prog_inds = prog_inds[prog_inds > 0]
        if len(prog_inds) == 0: continue

        # correct halo indices
        for k in range(len(prog_inds)):
            prog_ind = prog_inds[k]
            idx = (prog_ind % slab_factor)
            slab_id = ((prog_ind % id_factor - idx) // slab_factor)
            for i in range(len(slabs_prev)):
                slab_prev = slabs_prev[i]
                if slab_id == slab_prev:
                    idx += offsets[i]

            if (masses_prev[idx] < m_high) and (masses_prev[idx] >= m_low) and (idx != mp):
                N_merger[j] += 1
        
        #if num > 1: print(halo_inds, halo_ind_prev[main_progs[j]])
    return N_merger

def simple_load(filenames, fields):
    if type(filenames) is str:
        filenames = [filenames]
    
    do_prog = 'Progenitors' in fields
    
    Ntot = 0
    dtypes = {}
    
    if do_prog:
        N_prog_tot = 0
        fields.remove('Progenitors')  # treat specially
    
    for fn in filenames:
        with asdf.open(fn) as af:
            # Peek at the first field to get the total length
            # If the lengths of fields don't match up, that will be an error later
            Ntot += len(af['data'][fields[0]])
            
            for field in fields:
                if field not in dtypes:
                    if 'Position' == field:
                        dtypes[field] = (af['data'][field].dtype,3)
                    else:
                        dtypes[field] = af['data'][field].dtype
            
            if do_prog:
                N_prog_tot += len(af['data']['Progenitors'])
                
    # Make the empty tables
    t = Table({f:np.empty(Ntot, dtype=dtypes[f]) for f in fields}, copy=False)
    if do_prog:
        p = Table({'Progenitors':np.empty(N_prog_tot, dtype=np.int64)}, copy=False)
    
    # Fill the data into the empty tables
    j = 0
    jp = 0
    for i, fn in enumerate(filenames):
        #print(f"File number {i+1:d} of {len(filenames)}")
        f = asdf.open(fn, lazy_load=True, copy_arrays=True)
        fdata = f['data']
        thisN = len(fdata[fields[0]])
        
        for field in fields:
            # Insert the data into the next slot in the table
            t[field][j:j+thisN] = fdata[field]
            
        if do_prog:
            thisNp = len(fdata['Progenitors'])
            p['Progenitors'][jp:jp+thisNp] = fdata['Progenitors']
            jp += thisNp
            
        j += thisN
    
    # Should have filled the whole table!
    assert j == Ntot
    
    ret = dict(merger=t)
    if do_prog:
        ret['progenitors'] = p
        assert jp == N_prog_tot
        fields.append('Progenitors')
        
    return ret

def get_halos_per_slab(filenames, minified):
    # extract all slabs
    if minified:
        slabs = np.array([extract_superslab_minified(fn) for fn in filenames])
    else:
        slabs = np.array([extract_superslab(fn) for fn in filenames])
    n_slabs = len(slabs)
    N_halos_slabs = np.zeros(n_slabs, dtype=int)

    # extract number of halos in each slab
    for i in range(len(filenames)):
        fn = filenames[i]
        #print("File number %i of %i" % (i, len(filenames) - 1))
        f = asdf.open(fn)
        N_halos = f["data"]["HaloIndex"].shape[0]
        N_halos_slabs[i] = N_halos
        f.close()
        
    # sort in slab order
    i_sort = np.argsort(slabs)
    slabs = slabs[i_sort]
    N_halos_slabs = N_halos_slabs[i_sort]

    return N_halos_slabs, slabs

def extract_origin(fn):
    '''
    Extract index of the light cone origin
    example: 'Merger_lc1.02.asdf' should return 1
    '''
    origin = int(fn.split('lc')[-1].split('.')[0])
    return origin
