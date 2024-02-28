"""Config template for SSSA training and testing"""
from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
cfg = CN()


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
cfg.DATA = CN()

# path to the root of the dataset directory
cfg.DATA.path = ""
# dataset name
cfg.DATA.set = ""
# max number of objects in a video
cfg.DATA.max_objs = 10
# video fps
cfg.DATA.fps = 12
# video size
cfg.DATA.fpv = 24
# video height
cfg.DATA.height = 256
# video width
cfg.DATA.width = 256


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
cfg.TRAIN = CN()

# learn self-supervised
cfg.TRAIN.self_sup = True
# number of validation videos
cfg.TRAIN.val_count = 5
# number of frames overlapping per window 
cfg.TRAIN.stride = 1
# number of time steps to forecast
cfg.TRAIN.forecast = 3


cfg.TRAIN.OUTPUT = CN()
# directory to save training outputs
cfg.TRAIN.OUTPUT.path = ""

cfg.TRAIN.OUTPUT.FN = CN()
# weights filename
cfg.TRAIN.OUTPUT.FN.weight = "weights.pt"
# history filename
cfg.TRAIN.OUTPUT.FN.history = "history.csv"
# config filename
cfg.TRAIN.OUTPUT.FN.config = "config.yaml"
# log filename
cfg.TRAIN.OUTPUT.FN.log = "log.txt"


cfg.TRAIN.GPU = CN()
# buffer size for shuffle. Need to figure out limit.
cfg.TRAIN.GPU.shuffle_buffer_size = 1000
# number of GPUs to use for training. Only 1 GPU is supported for now.
cfg.TRAIN.GPU.count = 1
# number of videos per gradient update per GPU. Only batch size 1 is supported for now.
cfg.TRAIN.GPU.batch_size = 1
# GPU model
cfg.TRAIN.GPU.model = "NVIDIA GeForce RTX A4000"


cfg.TRAIN.STAGE_1 = CN()
# whether to include stage 1
cfg.TRAIN.STAGE_1.include = True
# number of epochs for stage 1
cfg.TRAIN.STAGE_1.num_epoch = 100


cfg.TRAIN.STAGE_2 = CN()
# whether to include stage 2
cfg.TRAIN.STAGE_2.include = True
# number of epochs for stage 2
cfg.TRAIN.STAGE_2.num_epoch = 100
# samples per pixel for renderer
cfg.TRAIN.STAGE_2.spp = 4
# pixel-wise MSE loss weight
cfg.TRAIN.STAGE_2.weight_mse = 1.0


cfg.TRAIN.OPTIM = CN()
# algorithm to use for optimization
cfg.TRAIN.OPTIM.optim = "adam"
# learning rate for training
cfg.TRAIN.OPTIM.lr = 0.001
# beta1 for Adam optimizer
cfg.TRAIN.OPTIM.beta1 = 0.9
# beta2 for Adam optimizer
cfg.TRAIN.OPTIM.beta2 = 0.999
# L2 penalty (regularization term) parameter
cfg.TRAIN.OPTIM.weight_decay = 0.0


# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
cfg.TEST = CN()


cfg.TEST.OUTPUT = CN()
# directory to save testing outputs
cfg.TEST.OUTPUT.path = ""

cfg.TEST.OUTPUT.FN = CN()
# Object modelling results
cfg.TEST.OUTPUT.FN.obj_mod = "obj_mod.csv"
# Video prediction results
cfg.TEST.OUTPUT.FN.vid_pred = "vid_pred.csv"


cfg.TEST.OBJ_MOD = CN()
# evaluate object modelling
cfg.TEST.OBJ_MOD.include = True


cfg.TEST.VID_PRED = CN()
# evaluate video prediction
cfg.TEST.VID_PRED.include = True
# samples per pixel for renderer
cfg.TEST.VID_PRED.spp = 16


cfg.TEST.GPU = CN()
# number of GPUs to use for training. Only 1 GPU is supported for now.
cfg.TEST.GPU.count = 1
# number of videos per gradient update per GPU. Only batch size 1 is supported for now.
cfg.TEST.GPU.batch_size = 1
# GPU model
cfg.TEST.GPU.model = "NVIDIA GeForce GTX 1060"


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
cfg.MODEL = CN()

# number of frames in the input window
cfg.MODEL.window = 3

cfg.MODEL.INPUT = CN()
# include video
cfg.MODEL.INPUT.video = True
# include depth
cfg.MODEL.INPUT.depth = True
# include optical flow
cfg.MODEL.INPUT.opt_flow = True
# include mask, otherwise append masks as seperate features
cfg.MODEL.INPUT.mask = True


cfg.MODEL.ENCODE = CN()

# temporal encoding
cfg.MODEL.ENCODE.TEMP = CN()
# temporal encoding architecture
cfg.MODEL.ENCODE.TEMP.arch = 'resnet_18'
# dropout rate
cfg.MODEL.ENCODE.TEMP.dropout = 0.1
# activation function
cfg.MODEL.ENCODE.TEMP.activation = 'relu'
# batch normalization
cfg.MODEL.ENCODE.TEMP.batch_norm = True
# embedding size
cfg.MODEL.ENCODE.TEMP.embed_size = 512

# prior encoding
cfg.MODEL.ENCODE.PRIOR = CN()
# include camera position in prior encoding
cfg.MODEL.ENCODE.PRIOR.cam_pos = True
# include camera orientation in prior encoding
cfg.MODEL.ENCODE.PRIOR.cam_ori = True
# include object class in prior encoding
cfg.MODEL.ENCODE.PRIOR.obj_class = True
# number of object classes
cfg.MODEL.ENCODE.PRIOR.num_obj_class = -1
# temporal encoding architecture
cfg.MODEL.ENCODE.PRIOR.arch = 'mlp'
# number of hidden layers
cfg.MODEL.ENCODE.PRIOR.num_layers = 1
# hidden layer size
cfg.MODEL.ENCODE.PRIOR.hidden_size = 2048
# dropout rate
cfg.MODEL.ENCODE.PRIOR.dropout = cfg.MODEL.ENCODE.TEMP.dropout
# activation function
cfg.MODEL.ENCODE.PRIOR.activation = cfg.MODEL.ENCODE.TEMP.activation
# batch normalization
cfg.MODEL.ENCODE.PRIOR.batch_norm = cfg.MODEL.ENCODE.TEMP.batch_norm
# embedding size
cfg.MODEL.ENCODE.PRIOR.embed_size = cfg.MODEL.ENCODE.TEMP.embed_size


# interaction encoding
cfg.MODEL.ENCODE.INTER = CN()
# include ineraction encoding
cfg.MODEL.ENCODE.INTER.include = True
# interaction encoding architecture
cfg.MODEL.ENCODE.INTER.arch = 'transformer_encoder'
# number of sub-layers in the encoder
cfg.MODEL.ENCODE.INTER.num_layers = 1
# number of heads in the multi-head attention
cfg.MODEL.ENCODE.INTER.num_heads = 8
# dropout rate
cfg.MODEL.ENCODE.INTER.dropout = cfg.MODEL.ENCODE.TEMP.dropout
# hidden layer size
cfg.MODEL.ENCODE.INTER.hidden_size = cfg.MODEL.ENCODE.PRIOR.hidden_size
# embedding size
cfg.MODEL.ENCODE.INTER.embed_size = cfg.MODEL.ENCODE.TEMP.embed_size


# parameter decoding
cfg.MODEL.DECODE = CN()

# base of parameter decoding
cfg.MODEL.DECODE.BASE = CN()
# decoder architecture
cfg.MODEL.DECODE.BASE.arch = 'mlp'
# number of hidden layers
cfg.MODEL.DECODE.BASE.num_layers = 1
# hidden layer size
cfg.MODEL.DECODE.BASE.hidden_size = cfg.MODEL.ENCODE.PRIOR.hidden_size
# dropout rate
cfg.MODEL.DECODE.BASE.dropout = cfg.MODEL.ENCODE.TEMP.dropout
# activation function
cfg.MODEL.DECODE.BASE.activation = cfg.MODEL.ENCODE.TEMP.activation
# batch normalization
cfg.MODEL.DECODE.BASE.batch_norm = cfg.MODEL.ENCODE.TEMP.batch_norm
# differentiable categorical estimator
cfg.MODEL.DECODE.BASE.category_est = 'gumbel_softmax'


# continuous parameter decoding
cfg.MODEL.DECODE.PARAM = CN()

# scale decoding
cfg.MODEL.DECODE.PARAM.SIZE = CN()
cfg.MODEL.DECODE.PARAM.SIZE.include = True
cfg.MODEL.DECODE.PARAM.SIZE.continuous = True
cfg.MODEL.DECODE.PARAM.SIZE.num_class = -1

# position decoding
cfg.MODEL.DECODE.PARAM.POS = CN()
cfg.MODEL.DECODE.PARAM.POS.include = True
cfg.MODEL.DECODE.PARAM.POS.continuous = True
cfg.MODEL.DECODE.PARAM.POS.num_class = -1

# orientation decoding
cfg.MODEL.DECODE.PARAM.ORI = CN()
cfg.MODEL.DECODE.PARAM.ORI.include = True
cfg.MODEL.DECODE.PARAM.ORI.continuous = True
cfg.MODEL.DECODE.PARAM.ORI.num_class = -1

# linear velocity decoding
cfg.MODEL.DECODE.PARAM.LIN_VEL  = CN()
cfg.MODEL.DECODE.PARAM.LIN_VEL.include = True
cfg.MODEL.DECODE.PARAM.LIN_VEL.continuous = True
cfg.MODEL.DECODE.PARAM.LIN_VEL.num_class = -1

# angular velocity decoding
cfg.MODEL.DECODE.PARAM.ANG_VEL  = CN()
cfg.MODEL.DECODE.PARAM.ANG_VEL.include = True
cfg.MODEL.DECODE.PARAM.ANG_VEL.continuous = True
cfg.MODEL.DECODE.PARAM.ANG_VEL.num_class = -1

# density decoding
#cfg.MODEL.DECODE.PARAM.DENSITY  = CN()
#cfg.MODEL.DECODE.PARAM.DENSITY.include = True
#cfg.MODEL.DECODE.PARAM.DENSITY.continuous = True
#cfg.MODEL.DECODE.PARAM.DENSITY.num_class = -1

# friction decoding
#cfg.MODEL.DECODE.PARAM.FRICTION  = CN()
#cfg.MODEL.DECODE.PARAM.FRICTION.include = True
#cfg.MODEL.DECODE.PARAM.FRICTION.continuous = True
#cfg.MODEL.DECODE.PARAM.FRICTION.num_class = -1

# restitution decoding
#cfg.MODEL.DECODE.PARAM.RESTITUTION  = CN()
#cfg.MODEL.DECODE.PARAM.RESTITUTION.include = True
#cfg.MODEL.DECODE.PARAM.RESTITUTION.continuous = True
#cfg.MODEL.DECODE.PARAM.RESTITUTION.num_class = -1

# material decoding
cfg.MODEL.DECODE.PARAM.MATERIAL  = CN()
cfg.MODEL.DECODE.PARAM.MATERIAL.include = True
cfg.MODEL.DECODE.PARAM.MATERIAL.continuous = True
cfg.MODEL.DECODE.PARAM.MATERIAL.num_class = -1

# colour decoding
cfg.MODEL.DECODE.PARAM.COLOR  = CN()
cfg.MODEL.DECODE.PARAM.COLOR.include = True
cfg.MODEL.DECODE.PARAM.COLOR.continuous = True
cfg.MODEL.DECODE.PARAM.COLOR.num_class = -1