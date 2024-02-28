"""Dataset handling for MOVi"""

import os

import tensorflow_datasets as tfds

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import torch
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_rotation_6d

from sssa.config import cfg
import data.movi_a
import data.movi_b


# Constants for mapping configuration to features in MOVI A and B datasets
MOVI_A = 'movi_a'
MOVI_B = 'movi_b'

# Feature type constants
INPUT = 'input'
PRIOR = 'prior'
OUTPUT = 'output'
LABEL = 'label'

# Input feature type constants
VIDEO = 'video'
DEPTH = 'depth'
OPT_FLOW = 'opt_flow'
MASK = 'mask'

# Prior feature type constants
CAM_POS = 'cam_pos'
CAM_ORI = 'cam_ori'
OBJ_CLASS = 'obj_class'

# Output feature type constants
SIZE = 'size'
POS = 'pos'
ORI = 'ori'
LIN_VEL = 'lin_vel'
ANG_VEL = 'ang_vel'
MATERIAL = 'material'
COLOR = 'color'

# Additional feature type constants
CAM = 'cam'
META = 'meta'
BACKGROUND = 'background'

# Configuration to feature mapping for MOVI A and B datasets
CFG2FEAT = {}
CFG2FEAT[MOVI_A] = {
    INPUT : {
        VIDEO : VIDEO,
        DEPTH : DEPTH,
        OPT_FLOW : 'forward_flow',
        MASK : 'segmentations'
    },
    PRIOR : {
        CAM_POS : ['camera','positions'],
        CAM_ORI : ['camera', 'quaternions'],
        OBJ_CLASS : ['instances', 'shape_label'],
    },
    OUTPUT : {
        SIZE : ['instances', 'size_label'],
        POS : ['instances', 'positions'],
        ORI : ['instances', 'quaternions'],
        LIN_VEL : ['instances', 'velocities'],
        ANG_VEL : ['instances', 'angular_velocities'],
        MATERIAL :['instances', 'material_label'],
        COLOR : ['instances', 'color_label']
    },
}
CFG2FEAT[MOVI_B] = {
    INPUT : CFG2FEAT[MOVI_A][INPUT],
    PRIOR: CFG2FEAT[MOVI_A][PRIOR],
    OUTPUT : {
        SIZE : ['instances', 'scale'],
        POS : CFG2FEAT[MOVI_A][OUTPUT][POS],
        ORI : CFG2FEAT[MOVI_A][OUTPUT][ORI],
        LIN_VEL : CFG2FEAT[MOVI_A][OUTPUT][LIN_VEL],
        ANG_VEL : CFG2FEAT[MOVI_A][OUTPUT][ANG_VEL],
        MATERIAL : CFG2FEAT[MOVI_A][OUTPUT][MATERIAL],
        COLOR : ['instances', 'color']
    }
}

# Constants for feature formatting types
RANGE = 'range'
ANGLE = 'angle'
ONE_HOT = 'one_hot'

# Configuration constant for number of classes in a feature
NUM_CLASS = 'num_class'

# Configuration for feature formatting for MOVI A and B datasets
FORMAT_FEATS = {}
FORMAT_FEATS[MOVI_A] = {
    INPUT : {
        DEPTH : RANGE,
        OPT_FLOW : RANGE
    },
    PRIOR : {
        CAM_ORI : ANGLE,
        OBJ_CLASS : ONE_HOT,
    },
    OUTPUT : {
        ORI : ANGLE,
        MATERIAL : ONE_HOT,
        COLOR : ONE_HOT
    }
}
FORMAT_FEATS[MOVI_B] = {
    INPUT : FORMAT_FEATS[MOVI_A][INPUT],
    PRIOR : FORMAT_FEATS[MOVI_A][PRIOR],
    OUTPUT : {
        ORI : FORMAT_FEATS[MOVI_A][OUTPUT][ORI],
        MATERIAL : FORMAT_FEATS[MOVI_A][OUTPUT][MATERIAL],
    }
}

# Constants for sequential features in MOVI A and B datasets
SEQ_FEATS = {}
SEQ_FEATS[MOVI_A] = {
     INPUT : [VIDEO, DEPTH, OPT_FLOW, MASK], 
     PRIOR : [CAM_POS, CAM_ORI],
     OUTPUT : [POS, ORI, LIN_VEL, ANG_VEL]
}
SEQ_FEATS[MOVI_B] = SEQ_FEATS[MOVI_A]

# Constants for features forecasted by training stage
STAGE_1 = 'stage_1'
STAGE_2 = 'stage_2'

FORECAST_FEATS = {
    STAGE_1 : [DEPTH, MASK],
    STAGE_2 : [VIDEO, MASK]
}


def load_train_data(cfg):
    """
    Loads and returns the training and validation datasets based on the provided configuration.

    Args:
        cfg: A configuration object containing dataset and training parameters.

    Returns:
        A tuple containing the training and validation datasets and dataset information.
    """
     # remove when running with full dataset
    full_train_count = 19
    assert full_train_count > cfg.TRAIN.val_count, "Validation count must be less than full train count"
    train_count = full_train_count - cfg.TRAIN.val_count

    split = [f'train[:{train_count}]', f'train[{train_count}:{full_train_count}]']
    (train_ds, val_ds), ds_info = tfds.load(cfg.DATA.set, data_dir=cfg.DATA.path,
                                            split=split, 
                                            shuffle_files=True,
                                            with_info=True)

    train_ds = train_ds.shuffle(cfg.TRAIN.GPU.shuffle_buffer_size, reshuffle_each_iteration=True)

    train_ds = tfds.as_numpy(train_ds)
    val_ds = tfds.as_numpy(val_ds)

    return (train_ds, val_ds), ds_info


def load_test_data(cfg):
    """
    Loads and returns the test dataset based on the provided configuration.

    Args:
        cfg: A configuration object containing dataset parameters.

    Returns:
        The test dataset and its information.
    """
    # remove when running with full dataset
    test_count = 8
    split = f'test[:{test_count}]'
    test_ds, ds_info = tfds.load(cfg.DATA.set, data_dir=cfg.DATA.path,
                                 split=split, 
                                 shuffle_files=False,
                                 with_info=True)

    test_ds = tfds.as_numpy(test_ds)

    return test_ds, ds_info


def get_feat(feat, sample):
    """
    Gets a feature from the sample dictionary based on the provided key.

    Args:
        feat: The key of the feature to be retrieved.
        sample: The sample dictionary from which the feature is extracted.
    
    Returns:
        The feature value from the sample dictionary.
    """
    if isinstance(feat, list):
        nested_sample = sample
        for key in feat:
            nested_sample = nested_sample[key]
        feat_val = nested_sample
    else:
        feat_val = sample[feat]
    
    return feat_val


def filter_sample(cfg, orig_sample, sample):
    """
    Filters the original sample based on the configuration and updates the provided sample dictionary.

    Args:
        cfg: A configuration object specifying which features to include.
        orig_sample: The original sample dictionary to be filtered.
        sample: The sample dictionary to be updated based on the filter.

    Returns:
        The updated sample dictionary after applying the filter.
    """
    for pair in CFG2FEAT[cfg.DATA.set][INPUT].items():
        # always include mask
        if cfg.MODEL.INPUT[pair[0]] or (pair[0] == MASK):
             sample[INPUT][pair[0]] = get_feat(pair[1], orig_sample)

    for pair in CFG2FEAT[cfg.DATA.set][PRIOR].items():
        if cfg.MODEL.ENCODE.PRIOR[pair[0]]:
            sample[PRIOR][pair[0]] = get_feat(pair[1], orig_sample)

    for pair in CFG2FEAT[cfg.DATA.set][OUTPUT].items():
        if cfg.MODEL.DECODE.PARAM[pair[0].upper()]:
            sample[OUTPUT][pair[0]] = get_feat(pair[1], orig_sample)
    
    return sample


def format_feat(feat, format, *args):
    """
    Formats a feature based on the specified format and additional arguments.

    Args:
        feat: The feature to be formatted.
        format: The format type (e.g., ANGLE, ONE_HOT, RANGE).
        *args: Additional arguments required for formatting.

    Returns:
        The formatted feature.
    """
    if format == ANGLE:
        quaternions = torch.tensor(feat, requires_grad=False)
        rotations = quaternion_to_matrix(quaternions)
        return matrix_to_rotation_6d(rotations).detach().numpy()

    elif format == ONE_HOT:
        classes = np.eye(args[0])[feat]
        return classes
    
    elif format == RANGE:
        return feat / 65535 * (args[1] - args[0]) + args[0]
    

def format_sample(cfg, orig_sample, sample):
    """
    Formats the sample based on the provided configuration and original sample.

    Args:
        cfg: A configuration object specifying formatting details.
        orig_sample: The original sample dictionary.
        sample: The sample dictionary to be formatted.

    Returns:
        The formatted sample dictionary.
    """
    # format input features
    for feat in FORMAT_FEATS[cfg.DATA.set][INPUT].keys():
        is_included = cfg.MODEL.INPUT[feat]
        is_range = FORMAT_FEATS[cfg.DATA.set][INPUT][feat] == RANGE

        if is_included and is_range:
            feat_name = CFG2FEAT[cfg.DATA.set][INPUT][feat]
            minv, maxv = orig_sample['metadata'][f'{feat_name}_range']
            sample[INPUT][feat] = format_feat(sample[INPUT][feat], RANGE, minv, maxv)  
        elif not is_range:
            raise NotImplementedError(f'{feat} formatting only implemented for {RANGE}')
    
    # format prior features
    for feat in FORMAT_FEATS[cfg.DATA.set][PRIOR].keys():
        is_included = cfg.MODEL.ENCODE.PRIOR[feat]
        is_formatted = feat in FORMAT_FEATS[cfg.DATA.set][PRIOR].keys()
        is_one_hot = is_formatted and (FORMAT_FEATS[cfg.DATA.set][PRIOR][feat] == ONE_HOT)
        is_range = is_formatted and (FORMAT_FEATS[cfg.DATA.set][PRIOR][feat] == RANGE)

        if is_included and is_one_hot:
            if feat == OBJ_CLASS:
                num_classes = cfg.MODEL.ENCODE.PRIOR[f'num_{OBJ_CLASS}'] + 1
                sample[PRIOR][feat] = format_feat(sample[PRIOR][feat] + 1, ONE_HOT, num_classes)
            else:
                num_classes = cfg.MODEL.ENCODE.PRIOR[f'num_{feat}']
                sample[PRIOR][feat] = format_feat(sample[PRIOR][feat], ONE_HOT, num_classes)

        elif is_included and is_range:
            raise NotImplementedError(f'{feat} formatting not implemented for {RANGE}')
        elif is_included and is_formatted:
            format = FORMAT_FEATS[cfg.DATA.set][PRIOR][feat]
            sample[PRIOR][feat] = format_feat(sample[PRIOR][feat], format)

    # format output features
    for feat in FORMAT_FEATS[cfg.DATA.set][OUTPUT].keys():
        is_included = cfg.MODEL.DECODE.PARAM[feat.upper()]
        is_range = FORMAT_FEATS[cfg.DATA.set][OUTPUT][feat] == RANGE
        is_one_hot = FORMAT_FEATS[cfg.DATA.set][OUTPUT][feat] == ONE_HOT

        if is_included and is_one_hot:
            format = FORMAT_FEATS[cfg.DATA.set][OUTPUT][feat]
            num_classes = cfg.MODEL.DECODE.PARAM[feat.upper()][NUM_CLASS]   
            sample[OUTPUT][feat] = format_feat(sample[OUTPUT][feat], ONE_HOT, num_classes)
        elif is_included and is_range:
            raise NotImplementedError(f'{feat} formatting not implemented for {RANGE}')
        elif is_included:
            format = FORMAT_FEATS[cfg.DATA.set][OUTPUT][feat]
            sample[OUTPUT][feat] = format_feat(sample[OUTPUT][feat], format)

    return sample


def window_seq_feat(cfg, feat, axis=0):
    """
    Applies windowing to a sequential feature based on the configuration.

    Args:
        cfg: A configuration object specifying windowing details.
        feat: The sequential feature to be windowed.
        axis: The axis along which to apply the windowing.

    Returns:
        A tuple containing the windowed values and labels.
    """
    if cfg.TRAIN.self_sup:
        window = cfg.MODEL.window + cfg.TRAIN.forecast
    else:
        window = cfg.MODEL.window

    sliding_windows = sliding_window_view(feat, window_shape=(window,), axis=axis)
    sliding_windows = np.moveaxis(sliding_windows, -1, axis+1)

    windows = sliding_windows.take(indices=range(0, sliding_windows.shape[axis], cfg.TRAIN.stride), axis=axis)
    vals = windows.take(indices=range(0, cfg.MODEL.window), axis=axis+1)
    labels = windows.take(indices=range(window - 1, window), axis=axis+1)

    return vals, labels


def window_const_feat(cfg, feat, axis=0):
    """
    Applies windowing to a constant feature based on the configuration.

    Args:
        cfg: A configuration object specifying windowing details.
        feat: The constant feature to be windowed.
        axis: The axis along which to apply the windowing.

    Returns:
        The windowed values of the constant feature.
    """    
    forecast_size = cfg.TRAIN.forecast if cfg.TRAIN.self_sup else 0
    num_windows = (cfg.DATA.fpv - cfg.MODEL.window - forecast_size) // cfg.TRAIN.stride + 1

    vals = np.repeat(np.expand_dims(feat, axis=axis) , num_windows, axis=axis)

    return vals


def window_sample(cfg, sample):
    """
    Applies windowing to the entire sample based on the configuration.

    Args:
        cfg: A configuration object specifying windowing details.
        sample: The sample dictionary to be windowed.

    Returns:
        The windowed sample dictionary.
    """
    sample[LABEL] = {}
    
    # input and related label windowing
    if cfg.TRAIN.self_sup:
        for feat in sample[INPUT].keys():
            stage_1_feat = cfg.TRAIN.STAGE_1.include and (feat in FORECAST_FEATS['stage_1'])
            stage_2_feat = cfg.TRAIN.STAGE_2.include and (feat in FORECAST_FEATS['stage_2'])
            
            feat_vals, feat_labels = window_seq_feat(cfg, sample[INPUT][feat])
            if stage_1_feat or stage_2_feat:
                sample[INPUT][feat] = feat_vals
                sample[LABEL][feat] = np.squeeze(feat_labels, axis=1)
            else:
                sample[INPUT][feat] = feat_vals

    else:
        for feat in sample[INPUT].keys():
            sample[INPUT][feat] = window_seq_feat(cfg, sample[INPUT][feat])[0]

        for feat in sample[OUTPUT].keys():
            if feat in SEQ_FEATS[cfg.DATA.set][OUTPUT]:
                windows = window_seq_feat(cfg, sample[OUTPUT][feat], axis=1)[1]
                windows = np.moveaxis(windows, 1, 0)
                sample[LABEL][feat] = np.squeeze(windows)
            else:
                sample[LABEL][feat] = window_const_feat(cfg, sample[OUTPUT][feat], axis=0)
    
    # priors windowing 
    for feat in sample[PRIOR].keys():
        if feat in SEQ_FEATS[cfg.DATA.set][PRIOR]:
            sample[PRIOR][feat] = window_seq_feat(cfg, sample[PRIOR][feat])[0]
            # if just want constant for window
            sample[PRIOR][feat] = np.take(sample[PRIOR][feat], 0, axis=1)
        else:
            sample[PRIOR][feat] = window_const_feat(cfg, sample[PRIOR][feat])

    return sample


def process_sample(cfg, orig_sample):
    """
    Processes a single sample from the dataset based on the provided configuration.

    Args:
        cfg: A configuration object detailing how the sample should be processed.
        orig_sample: The original sample to be processed.

    Returns:
        A processed sample dictionary.
    """
    sample = {
         INPUT : {},
         PRIOR : {},
         OUTPUT: {}
    }

    sample = filter_sample(cfg, orig_sample, sample)
    sample = format_sample(cfg, orig_sample, sample)
    sample = window_sample(cfg, sample)
    
    return sample    


def add_misc_details(orig_sample, sample):
    """
    Adds miscellaneous details from the original sample to the new sample based on the configuration.

    Args:
        cfg: A configuration object.
        orig_sample: The original sample dictionary.
        sample: The new sample dictionary to which details are added.

    Returns:
        The new sample dictionary with added miscellaneous details.
    """
    sample[CAM] = {}
    sample[META] = {}

    for k, v in orig_sample['camera'].items():
        if k not in ['positions', 'quaternions']:
            sample[CAM][k] = v

    for k, v in orig_sample['metadata'].items():
        if k not in ['depth_range', 'forward_flow_range', 'backward_flow_range']:
            sample[META][k] = v
    
    if cfg.DATA.set == MOVI_B:
        sample[BACKGROUND][COLOR] = orig_sample[BACKGROUND + '_' + COLOR] 
    
    return sample


def mask_input(cfg, sample):
    """
    Masks the input features based on the configuration.

    Args:
        cfg: A configuration object specifying which input features to mask.
        sample: The sample dictionary to be masked.

    Returns:
        The masked sample dictionary.
    """
    # TODO: implement masking
    pass


def finalize_sample(cfg, orig_sample, sample):
    """
    Finalizes the sample by adding miscellaneous details and masking input features.
    
    Args:
        cfg: A configuration object detailing how the sample should be finalized.
        orig_sample: The original sample dictionary.
        sample: The sample dictionary to be finalized.
    
    Returns:
        The finalized sample dictionary.
    """

    # cleanup
    del sample[OUTPUT]

    # final touches
    add_misc_details(orig_sample, sample)
    mask_input(cfg, sample)

    return sample


#################
# Sanity checks #
#################

def print_sample(sample):
    """
    Prints the contents of the sample dictionary.

    Args:
        sample: The sample dictionary to be printed.
    """
    for name in sample.keys():
        print(name)
        if isinstance(sample[name], dict):
            for k, v in sample[name].items():
                    print(k, v.shape)
        else:
            print(sample[name])
        print()


def check_seq_feat(cfg, feat_name, orig_feat, feat_windows):
    """
    Checks if sequential feature windowing is correctly done.

    Args:
        cfg: Configuration object with model and training parameters.
        orig_feat: the original feature
        feat_windows: Processed sample windowed features
    """
    forecast_size = cfg.TRAIN.forecast if cfg.TRAIN.self_sup else 0
    num_windows = (cfg.DATA.fpv - cfg.MODEL.window - forecast_size) // cfg.TRAIN.stride + 1

    assert feat_windows.shape[0] == num_windows, f"{feat_name} sequential feature mismatched window size: {feat_windows.shape[0]} != {num_windows}"

    for i in range(num_windows):
        start_index = i * cfg.TRAIN.stride
        end_index = start_index + cfg.MODEL.window
        orig_window = orig_feat[start_index:end_index]
        feat_window = feat_windows[i]

        assert np.all(orig_window == feat_window), f"{feat_name} sequential feature windows are not equal"


def check_const_feat(cfg, feat_name, orig_feat, feat_windows):
    """
    Checks if constant feature windowing is correctly done.

    Args:
        cfg: Configuration object with model and training parameters.
        orig_feat: the original feature
        feat_windows: Processed sample windowed features
    """
    forecast_size = cfg.TRAIN.forecast if cfg.TRAIN.self_sup else 0
    num_windows = (cfg.DATA.fpv - cfg.MODEL.window - forecast_size) // cfg.TRAIN.stride + 1

    assert feat_windows.shape[0] == num_windows, f"{feat_name} constant feature mismatched window size: {feat_windows.shape[0]} != {num_windows}"

    for i in range(num_windows):
        orig_window = orig_feat
        feat_window = feat_windows[i]

        assert np.all(orig_window == feat_window), f"{feat_name} constant feature windows are not equal"


def check_seq_label_feat(cfg, feat_name, orig_feat, feat_windows):
    """
    Checks if sequential label feature windowing is correctly done.

    Args:
        cfg: Configuration object with model and training parameters.
        orig_feat: the original feature
        feat_windows: Processed sample windowed features
    """
    forecast_size = cfg.TRAIN.forecast if cfg.TRAIN.self_sup else 0
    num_windows = (cfg.DATA.fpv - cfg.MODEL.window - forecast_size) // cfg.TRAIN.stride + 1

    assert feat_windows.shape[0] == num_windows, f"{feat_name} sequential label feature mismatched window size: {feat_windows.shape[0]} != {num_windows}"

    for i in range(num_windows):
        start_index = i * cfg.TRAIN.stride
        orig_window = orig_feat[start_index + cfg.MODEL.window + forecast_size - 1]
        feat_window = feat_windows[i]

        assert np.all(orig_window == feat_window), f"{feat_name} sequential label feature windows are not equal"


def check_const_label_feat(cfg, feat_name, orig_feat, feat_windows):
    """
    Checks if constant label feature windowing is correctly done.

    Args:
        cfg: Configuration object with model and training parameters.
        orig_feat: the original feature
        feat_windows: Processed sample windowed features
    """
    forecast_size = cfg.TRAIN.forecast if cfg.TRAIN.self_sup else 0
    num_windows = (cfg.DATA.fpv - cfg.MODEL.window - forecast_size) // cfg.TRAIN.stride + 1

    assert feat_windows.shape[0] == num_windows, f"{feat_name} constant label feature mismatched window size: {feat_windows.shape[0]} != {num_windows}"

    for i in range(num_windows):
        orig_window = orig_feat
        feat_window = feat_windows[i]

        assert np.all(orig_window == feat_window), f"{feat_name} constant label feature windows are not equal"


def sanity_check(cfg):
    """
    Checks if the sample has been processed correctly.

    Args:
        cfg: Configuration object with model and training parameters.
        sample: The sample dictionary to be checked.
    """
    print(f'Testing {cfg.DATA.set} dataset processing')

    (train_ds, _), _ = load_train_data(cfg)
    example = next(iter(train_ds))

    for self_sup in [True, False]:
        cfg.TRAIN.self_sup = self_sup
        sample = process_sample(cfg, example)

        for feat in sample[INPUT].keys():
            is_seq = feat in SEQ_FEATS[cfg.DATA.set][INPUT]
            is_formatted = feat in FORMAT_FEATS[cfg.DATA.set][INPUT].keys()
            is_ranged = is_formatted and (FORMAT_FEATS[cfg.DATA.set][INPUT][feat] == RANGE)
            
            feat_name = CFG2FEAT[cfg.DATA.set][INPUT][feat]
            orig_feat = get_feat(feat_name, example)

            if is_seq and is_ranged:
                minv, maxv = example['metadata'][f'{feat_name}_range']
                orig_feat = format_feat(orig_feat, RANGE, minv, maxv)
            elif is_seq and not is_formatted:
                pass
            else:
                raise NotImplementedError(f'{feat} input feature not implemented for testing')
            
            if is_seq:
                check_seq_feat(cfg, feat, orig_feat, sample[INPUT][feat])

    print('(1/3)\tPassed Input Testing!')

    for self_sup in [True, False]:
        cfg.TRAIN.self_sup = self_sup
        sample = process_sample(cfg, example)
        
        for feat in sample[PRIOR].keys():
            is_formatted = feat in FORMAT_FEATS[cfg.DATA.set][PRIOR].keys()
            is_one_hot = is_formatted and (FORMAT_FEATS[cfg.DATA.set][PRIOR][feat] == ONE_HOT)
            is_range = is_formatted and (FORMAT_FEATS[cfg.DATA.set][PRIOR][feat] == RANGE)

            feat_name  = CFG2FEAT[cfg.DATA.set][PRIOR][feat]
            orig_feat = get_feat(feat_name, example)

            if is_one_hot:
                format = FORMAT_FEATS[cfg.DATA.set][PRIOR][feat],
                if feat == OBJ_CLASS:
                    num_classes = cfg.MODEL.ENCODE.PRIOR[f'num_{OBJ_CLASS}'] + 1
                    orig_feat = format_feat(orig_feat + 1, ONE_HOT, num_classes)
                else:
                    num_classes = cfg.MODEL.ENCODE.PRIOR[f'num_{feat}']
                    orig_feat = format_feat(orig_feat, ONE_HOT, num_classes)
            elif is_range:
                minv, maxv = example['metadata'][f'{feat_name}_range']
                orig_feat = format_feat(orig_feat, RANGE, minv, maxv)
            elif is_formatted:
                format = FORMAT_FEATS[cfg.DATA.set][PRIOR][feat]
                orig_feat = format_feat(orig_feat, format)   
            
            if feat in SEQ_FEATS[cfg.DATA.set][PRIOR]:
                check_seq_feat(cfg, feat, orig_feat, sample[PRIOR][feat])
            else:
                check_const_feat(cfg, feat, orig_feat, sample[PRIOR][feat])

    print('(2/3)\tPassed Priors Testing!')

    for self_sup in [True, False]:
            cfg.TRAIN.self_sup = self_sup
            sample = process_sample(cfg, example)

            if self_sup:
                for feat in sample[LABEL].keys():
                    feat_name = CFG2FEAT[cfg.DATA.set][INPUT][feat]
                    orig_feat = get_feat(feat_name, example)

                    is_seq = feat in SEQ_FEATS[cfg.DATA.set][INPUT]
                    is_formatted = feat in FORMAT_FEATS[cfg.DATA.set][INPUT].keys()
                    is_ranged = is_formatted and (FORMAT_FEATS[cfg.DATA.set][INPUT][feat] == RANGE)
                    
                    if is_seq and is_ranged:
                        minv, maxv = example['metadata'][f'{feat_name}_range']
                        orig_feat = format_feat(orig_feat, RANGE, minv, maxv)
                    elif is_seq and not is_formatted:
                        pass
                    else:
                        raise NotImplementedError(f'{feat} input feature not implemented for testing')
                    
                    if is_seq:
                        check_seq_label_feat(cfg, feat, orig_feat, sample[LABEL][feat])

            else:
                for feat in sample[LABEL].keys():
                    feat_name = CFG2FEAT[cfg.DATA.set][OUTPUT][feat]
                    orig_feat = get_feat(feat_name, example)
                    
                    is_formatted = feat in FORMAT_FEATS[cfg.DATA.set][OUTPUT].keys()
                    is_range = is_formatted and (FORMAT_FEATS[cfg.DATA.set][OUTPUT][feat] == RANGE)
                    is_one_hot = is_formatted and (FORMAT_FEATS[cfg.DATA.set][OUTPUT][feat] == ONE_HOT)

                    if is_one_hot:
                        num_classes = cfg.MODEL.DECODE.PARAM[feat.upper()][NUM_CLASS]   
                        orig_feat = format_feat(orig_feat, ONE_HOT, num_classes)
                    elif is_range:
                        minv, maxv = example['metadata'][f'{feat_name}_range']
                        orig_feat = format_feat(orig_feat, RANGE, minv, maxv)
                    elif is_formatted:
                        format = FORMAT_FEATS[cfg.DATA.set][OUTPUT][feat]
                        orig_feat = format_feat(orig_feat, format)

                    if feat in SEQ_FEATS[cfg.DATA.set][OUTPUT]:
                        # kinda treat as constants
                        orig_feat = np.moveaxis(orig_feat, 1, 0)
                        check_seq_label_feat(cfg, feat, orig_feat, sample[LABEL][feat])
                    else:
                        check_const_label_feat(cfg, feat, orig_feat, sample[LABEL][feat])

    print('(3/3)\tPassed Label Testing!')


if __name__ == '__main__':
    """Checking dataset processing has formatted and windowed features correctly."""

    project_dp = os.path.expanduser('~/git/sssa-object-modelling/')
    for set_name in [MOVI_A]:
        cfg.DATA.set = set_name
        config_fp = os.path.join(project_dp, f'cfg/templates/sssa-{set_name}.yaml')
        cfg.merge_from_file(config_fp)
        cfg.DATA.path = os.path.join(project_dp, 'data/sets/')
        sanity_check(cfg)


                
