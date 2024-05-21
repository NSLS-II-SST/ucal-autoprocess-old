import mass
import numpy as np
import matplotlib.pyplot as plt
from ucalpost.databroker.run import get_tes_state, get_filename, get_logname, get_line_names
from ucalpost.databroker.catalog import WrappedDatabroker
from ucalpost.tes.process_classes import log_from_run, ProcessedData, ScanData
from tiled.client import from_profile
import calibration


plt.close("all")
plt.ion()

from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks

c = WrappedDatabroker(from_profile("ucal")).filter_by_stop()

def get_model_file(run):
    return "/home/decker/data/20240507/0000/20240507_run0000_model.hdf5"

def process_state(data, state, model_path=None):
    if model_path is not None:
        data.add5LagRecipes(model_path)
        infix = "5Lag"
    else:
        infix = ""
    data.learnDriftCorrection(indicatorName="pretriggerMean", uncorrectedName=f"filtValue{infix}", 
                          correctedName=f"filtValue{infix}DC", states=state) 
    data.learnDriftCorrection(indicatorName="filtPhase", uncorrectedName=f"filtValue{infix}DC", 
                          correctedName=f"filtValue{infix}DCPC", states=state)
    fv_attr = f"filtValue{infix}DCPC"
    return fv_attr

def process_and_calibrate_run(run):
    """
    Must take a calibration run
    """
    filename = get_filename(run, convert_local=False)
    files = getOffFileListFromOneFile(filename, maxChans=400)
    data = ChannelGroup(files)
    state = get_tes_state(run)
    model_path = get_model_file(run)
    fv_attr = process_state(data, state, model_path)

    line_names = get_line_names(run)
    data.calibrate(state, line_names, fv_attr)
    return data


def get_tes_arrays(data, state):
    timestamps = []
    energies = []
    channels = []
    for ds in data.values():
        try:
            uns, es = ds.getAttr(["unixnano", "energy"], state)
        except:
            print(f"{ds.channum} failed")
            ds.markBad("Failed to get energy")
            continue
        ch = np.zeros_like(uns) + ds.channum
        timestamps.append(uns*1e-9)
        energies.append(es)
        channels.append(ch)
    ts_arr = np.concatenate(timestamps)
    en_arr = np.concatenate(energies)
    ch_arr = np.concatenate(channels)
    sort_idx = np.argsort(ts_arr)
    
    timestamps=ts_arr[sort_idx]
    energies=en_arr[sort_idx]
    channels=ch_arr[sort_idx]
    
    return timestamps, energies, channels

def sd_from_run(data, run):
    state = get_tes_state(run)
    ts, en, ch = get_tes_arrays(data, state)
    sd = ScanData(ProcessedData(ts, en, ch), log_from_run(run))
    return sd

data = None

def handle_run(run):
    scantype = run.start.get('scantype', "")
    if scantype == "calibration":
        global data
        data = process_and_calibrate_run(run)
    elif scantype != "" and data is not None:
        sd = sd_from_run(data, run)
        filename = f"{run.start['plan_name']}_{run.start['scan_id']}.npz"
        rixs_filename = f"{run.start['plan_name']}_rixs_{run.start['scan_id']}.npz"
        tfy, mono = sd.getScan1d(200, 1000)
        i0 = run.primary.data['i0'].read()
        np.savez(filename, mono=mono, tfy=tfy, i0=i0)
        z, x, y = sd.getScan2d(200, 1000)
        np.savez(rixs_filename, mono=x, emission=y, counts=z)
    elif data is None:
        print("No data has been loaded yet! Need to process a calibration run for this file")
    elif scantype == "":
        print("Data has no scantype, is it a TES run?")

#####################################################################################
# Example of saving some data to disk after processing + calibration                #
#####################################################################################

cal_run = c['bad32542-ab64-44f4-9146-614aac4b17c1']

# First, process the calibration so that we have data loaded and calibrated to energy
handle_run(cal_run)

# Next, receive a stop document, get the run, send in the actual data, and save arrays to disk

def listen_for_stop_documents():
    stop_doc = {'uid': 'e6526ae5-07dc-46c2-a407-ca5543393516',
                'time': 1715171398.4238985,
                'run_start': 'ffe3541a-be03-4898-981c-5084026c27b5',
                'exit_status': 'success',
                'reason': '',
                'num_events': {'baseline': 2, 'primary': 436}}
    return stop_doc

#while True:
stop_doc = listen_for_stop_documents()
run = c[stop_doc['run_start']]
handle_run(run)


