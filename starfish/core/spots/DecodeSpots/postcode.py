from copy import deepcopy
from typing import Any, Hashable, Mapping, Tuple

import numpy as np
import pandas as pd

from starfish.core.codebook.codebook import Codebook
from starfish.core.imagestack.imagestack import ImageStack
from starfish.core.intensity_table.decoded_intensity_table import DecodedIntensityTable
from starfish.core.intensity_table.intensity_table import IntensityTable
from starfish.core.intensity_table.intensity_table_coordinates import \
    transfer_physical_coords_to_intensity_table
from starfish.core.spots.DecodeSpots.trace_builders import build_spot_traces_exact_match
from starfish.core.types import SpotFindingResults
from starfish.types import Axes, Features
from ._base import DecodeSpotsAlgorithm
from .postcode_funcs import decoding_function, decoding_output_to_dataframe

class postcodeDecode(DecodeSpotsAlgorithm):

    def __init__(self, codebook: Codebook):
        self.codebook = codebook

    def run(self, spots: SpotFindingResults):

        # Format starfish spots for use in postcode
        bd_table = build_spot_traces_exact_match(spots)
        spots_s = np.swapaxes(bd_table.data, 1, 2)
        spots_loc_s = pd.DataFrame(columns=['X', 'Y', 'Z'])
        spots_loc_s['X'] = np.array(bd_table.x)
        spots_loc_s['Y'] = np.array(bd_table.y)
        spots_loc_s['Z'] = np.array(bd_table.z)
        barcodes_01 = np.swapaxes(np.array(self.codebook), 1, 2)
        K = barcodes_01.shape[0]

        # Decode using postcode
        out = decoding_function(spots_s, barcodes_01, print_training_progress=True)

        # Reformat output into pandas dataframe
        df_class_names = np.concatenate((self.codebook.target.values,
                                         ['infeasible', 'background', 'nan']))
        barcodes_0123 = np.argmax(np.array(self.codebook), axis=2)
        channel_base = ['T', 'G', 'C', 'A']
        barcodes_AGCT = np.empty(K, dtype='object')
        for k in range(K):
            barcodes_AGCT[k] = ''.join(list(np.array(channel_base)[barcodes_0123[k, :]]))
        df_class_codes = np.concatenate((barcodes_AGCT, ['NA', '0000', 'NA']))
        decoded_spots_df = decoding_output_to_dataframe(out, df_class_names, df_class_codes)
        decoded_df_s = pd.concat([decoded_spots_df, spots_loc_s], axis=1)

        # Remove infeasible and background codes
        decoded_df_s = decoded_df_s[~np.isin(decoded_df_s['Name'], ['background', 'infeasible'])].reset_index(drop=True)

        # create empty IntensityTable filled with np.nan
        channels = spots.ch_labels
        rounds = spots.round_labels
        data = np.full((len(decoded_df_s), len(rounds), len(channels)), fill_value=np.nan)
        dims = (Features.AXIS, Axes.ROUND.value, Axes.CH.value)

        coords: Mapping[Hashable, Tuple[str, Any]] = {
            Features.SPOT_RADIUS: (Features.AXIS, np.full(len(decoded_df_s), 1)),
            Axes.ZPLANE.value: (Features.AXIS, np.asarray([z for z in decoded_df_s['Z']])),
            Axes.Y.value: (Features.AXIS, np.asarray([x for x in decoded_df_s['X']])),
            Axes.X.value: (Features.AXIS, np.asarray([y for y in decoded_df_s['Y']])),
            Features.SPOT_ID: (Features.AXIS, np.arange(len(decoded_df_s))),
            Features.AXIS: (Features.AXIS, np.arange(len(decoded_df_s))),
            Axes.ROUND.value: (Axes.ROUND.value, rounds),
            Axes.CH.value: (Axes.CH.value, channels)
        }
        int_table = IntensityTable(data=data, dims=dims, coords=coords)

        

        # Create dictionary of dictionaries where first dictionary's key is the (round, channel) tuple and the 
        # value is a dictionary with keys of spot coordinates as a string "z_y_x" with the intensity as the value
        # (allows for fast lookup of spot intensity from coordinates)
        spot_items = dict(spots.items())
        for rch in spot_items:
            spot_items[rch] = spot_items[rch].spot_attrs.data
            zs = spot_items[rch]['z']
            ys = spot_items[rch]['y']
            xs = spot_items[rch]['x']
            spot_items[rch].index = ['_'.join([str(zs[i]), str(ys[i]), str(xs[i])]) for i in range(len(spot_items[rch]))]
            spot_items[rch] = spot_items[rch]['intensity'].to_dict()

        # Fill in intensity values
        # X and Y are switch because postcode switches them at some point and this reswitches them
        zs = decoded_df_s['Z']
        ys = decoded_df_s['X']
        xs = decoded_df_s['Y']
        result_ints = []
        for i in range(len(decoded_df_s)):
            int_vector = np.zeros((len(self.codebook.r), len(self.codebook.c)), dtype='float32')
            for r in range(len(self.codebook.r)):
                for ch in range(len(self.codebook.c)):
                    int_vector[r, ch] = spot_items[(r, ch)]['_'.join([str(zs[i]), str(xs[i]), str(ys[i])])]
            result_ints.append(int_vector)
        int_table.values = np.array(result_ints)
        
        # Reswap x and y values
        tmp = deepcopy(int_table['y'].data)
        int_table['y'].data = deepcopy(int_table['x'].data)
        int_table['x'].data = tmp
        
        int_table = transfer_physical_coords_to_intensity_table(intensity_table=int_table,
                                                                spots=spots)

        # Create DecodedIntensityTable and return
        return DecodedIntensityTable.from_intensity_table(
            int_table,
            targets=(Features.AXIS, decoded_df_s['Name'].astype('U')))
