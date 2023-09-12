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

def torch_format(numpy_array):
    D = numpy_array.shape[1] * numpy_array.shape[2]
    return np.transpose(numpy_array, axes=[0,2,1]).reshape(numpy_array.shape[0], D)

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

        real_codes = self.codebook[['blank' not  in target.lower() for target in self.codebook['target'].data]]
        blank_codes = self.codebook[['blank' in target.lower() for target in self.codebook['target'].data]]
        barcodes_01 = np.swapaxes(np.array(real_codes), 1, 2)
        barcodes_02 = np.swapaxes(np.array(blank_codes), 1, 2)
        K = barcodes_01.shape[0] + barcodes_02.shape[0]

        # Decode using postcode
        out = decoding_function(spots_s, barcodes_01, print_training_progress=True)
        
        # Subset for just the codes included in codebook (current values are for all possible codes)
        keep = []
        for codebook_code in torch_format(np.swapaxes(np.array(self.codebook), 1, 2)):
            for i, possible_code in enumerate(out['class_ind']['codes']):
                if list(np.array(possible_code)) == list(codebook_code):
                    keep.append(i)
                    break
        shape = out['class_probs'].shape
        out['class_probs'] = out['class_probs'][:, keep + [shape[1]-1]]
        out['class_ind']['genes'] = np.array(range(len(self.codebook)))
        out['class_ind']['bkg'] = self.codebook.shape[0]
        out['class_ind']['inf'] = self.codebook.shape[0] + 1
        

        # Reformat output into pandas dataframe
        df_class_names = np.concatenate((self.codebook.target.values,
                                         ['background', 'nan']))
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
        print(Counter(['blank' in target for target in decoded_df_s['Name']]))

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
