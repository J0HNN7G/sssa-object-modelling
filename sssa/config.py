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
# number of time steps to forecast
cfg.TRAIN.STAGE_1.forecast = 3


cfg.TRAIN.STAGE_2 = CN()
# whether to include stage 2
cfg.TRAIN.STAGE_2.include = True
# number of epochs for stage 2
cfg.TRAIN.STAGE_2.num_epoch = cfg.TRAIN.STAGE_1.num_epoch
# number of time steps to forecast
cfg.TRAIN.STAGE_2.forecast = cfg.TRAIN.STAGE_1.forecast
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

cfg.MODEL.INPUT = CN()
# number of frames in the input window
cfg.MODEL.INPUT.window = 3
# include video
cfg.MODEL.INPUT.video = True
# include depth
cfg.MODEL.INPUT.depth = True
# include optical flow
cfg.MODEL.INPUT.opt_flow = True
# include mask, otherwise append masks as seperate features
cfg.MODEL.INPUT.mask = True


cfg.MODEL.ENCODING = CN()

# temporal encoding
cfg.MODEL.ENCODING.TEMPORAL = CN()
# temporal encoding architecture
cfg.MODEL.ENCODING.TEMPORAL.arch = 'resnet_18'
# dropout rate
cfg.MODEL.ENCODING.TEMPORAL.dropout = 0.1
# activation function
cfg.MODEL.ENCODING.TEMPORAL.activation = 'relu'
# batch normalization
cfg.MODEL.ENCODING.TEMPORAL.batch_norm = True
# embedding size
cfg.MODEL.ENCODING.TEMPORAL.embed_size = 512

# prior encoding
cfg.MODEL.ENCODING.PRIORS = CN()
# include camera pose in prior encoding
cfg.MODEL.ENCODING.PRIORS.cam_pose = True
# include object class in prior encoding
cfg.MODEL.ENCODING.PRIORS.obj_class = True
# number of object classes
cfg.MODEL.ENCODING.PRIORS.num_obj_class = -1
# temporal encoding architecture
cfg.MODEL.ENCODING.PRIORS.arch = 'mlp'
# number of hidden layers
cfg.MODEL.ENCODING.PRIORS.num_layers = 1
# hidden layer size
cfg.MODEL.ENCODING.PRIORS.hidden_size = 2048
# dropout rate
cfg.MODEL.ENCODING.PRIORS.dropout = cfg.MODEL.ENCODING.TEMPORAL.dropout
# activation function
cfg.MODEL.ENCODING.PRIORS.activation = cfg.MODEL.ENCODING.TEMPORAL.activation
# batch normalization
cfg.MODEL.ENCODING.PRIORS.batch_norm = cfg.MODEL.ENCODING.TEMPORAL.batch_norm
# embedding size
cfg.MODEL.ENCODING.PRIORS.embed_size = cfg.MODEL.ENCODING.TEMPORAL.embed_size


# interaction encoding
cfg.MODEL.ENCODING.INTERACTION = CN()
# include ineraction encoding
cfg.MODEL.ENCODING.INTERACTION.include = True
# interaction encoding architecture
cfg.MODEL.ENCODING.INTERACTION.arch = 'transformer_encoder'
# number of sub-layers in the encoder
cfg.MODEL.ENCODING.INTERACTION.num_layers = 1
# number of heads in the multi-head attention
cfg.MODEL.ENCODING.INTERACTION.num_heads = 8
# dropout rate
cfg.MODEL.ENCODING.INTERACTION.dropout = cfg.MODEL.ENCODING.TEMPORAL.dropout
# hidden layer size
cfg.MODEL.ENCODING.INTERACTION.hidden_size = cfg.MODEL.ENCODING.PRIORS.hidden_size
# embedding size
cfg.MODEL.ENCODING.INTERACTION.embed_size = cfg.MODEL.ENCODING.TEMPORAL.embed_size


# parameter decoding
cfg.MODEL.DECODING = CN()

# base of parameter decoding
cfg.MODEL.DECODING.BASE = CN()
# decoder architecture
cfg.MODEL.DECODING.BASE.arch = 'mlp'
# number of hidden layers
cfg.MODEL.DECODING.BASE.num_layers = 1
# hidden layer size
cfg.MODEL.DECODING.BASE.hidden_size = cfg.MODEL.ENCODING.PRIORS.hidden_size
# dropout rate
cfg.MODEL.DECODING.BASE.dropout = cfg.MODEL.ENCODING.TEMPORAL.dropout
# activation function
cfg.MODEL.DECODING.BASE.activation = cfg.MODEL.ENCODING.TEMPORAL.activation
# batch normalization
cfg.MODEL.DECODING.BASE.batch_norm = cfg.MODEL.ENCODING.TEMPORAL.batch_norm
# differentiable categorical estimator
cfg.MODEL.DECODING.BASE.category_est = 'gumbel_softmax'


# continuous parameter decoding
cfg.MODEL.DECODING.PARAMETERS = CN()

# scale decoding
cfg.MODEL.DECODING.PARAMETERS.SCALE = CN()
cfg.MODEL.DECODING.PARAMETERS.SCALE.include = True
cfg.MODEL.DECODING.PARAMETERS.SCALE.continuous = True
cfg.MODEL.DECODING.PARAMETERS.SCALE.num_category = -1

# position decoding
cfg.MODEL.DECODING.PARAMETERS.POSITION = CN()
cfg.MODEL.DECODING.PARAMETERS.POSITION.include = True
cfg.MODEL.DECODING.PARAMETERS.POSITION.continuous = True
cfg.MODEL.DECODING.PARAMETERS.POSITION.num_category = -1

# orientation decoding
cfg.MODEL.DECODING.PARAMETERS.ORIENTATION = CN()
cfg.MODEL.DECODING.PARAMETERS.ORIENTATION.include = True
cfg.MODEL.DECODING.PARAMETERS.ORIENTATION.continuous = True
cfg.MODEL.DECODING.PARAMETERS.ORIENTATION.num_category = -1

# linear velocity decoding
cfg.MODEL.DECODING.PARAMETERS.LINEAR_VELOCITY  = CN()
cfg.MODEL.DECODING.PARAMETERS.LINEAR_VELOCITY.include = True
cfg.MODEL.DECODING.PARAMETERS.LINEAR_VELOCITY.continuous = True
cfg.MODEL.DECODING.PARAMETERS.LINEAR_VELOCITY.num_category = -1

# angular velocity decoding
cfg.MODEL.DECODING.PARAMETERS.ANGULAR_VELOCITY  = CN()
cfg.MODEL.DECODING.PARAMETERS.ANGULAR_VELOCITY.include = True
cfg.MODEL.DECODING.PARAMETERS.ANGULAR_VELOCITY.continuous = True
cfg.MODEL.DECODING.PARAMETERS.ANGULAR_VELOCITY.num_category = -1

# density decoding
#cfg.MODEL.DECODING.PARAMETERS.DENSITY  = CN()
#cfg.MODEL.DECODING.PARAMETERS.DENSITY.include = True
#cfg.MODEL.DECODING.PARAMETERS.DENSITY.continuous = True
#cfg.MODEL.DECODING.PARAMETERS.DENSITY.num_category = -1

# friction decoding
#cfg.MODEL.DECODING.PARAMETERS.FRICTION  = CN()
#cfg.MODEL.DECODING.PARAMETERS.FRICTION.include = True
#cfg.MODEL.DECODING.PARAMETERS.FRICTION.continuous = True
#cfg.MODEL.DECODING.PARAMETERS.FRICTION.num_category = -1

# restitution decoding
#cfg.MODEL.DECODING.PARAMETERS.RESTITUTION  = CN()
#cfg.MODEL.DECODING.PARAMETERS.RESTITUTION.include = True
#cfg.MODEL.DECODING.PARAMETERS.RESTITUTION.continuous = True
#cfg.MODEL.DECODING.PARAMETERS.RESTITUTION.num_category = -1

# colour decoding
cfg.MODEL.DECODING.PARAMETERS.COLOUR  = CN()
cfg.MODEL.DECODING.PARAMETERS.COLOUR.include = True
cfg.MODEL.DECODING.PARAMETERS.COLOUR.continuous = True
cfg.MODEL.DECODING.PARAMETERS.COLOUR.num_category = -1

# material decoding
cfg.MODEL.DECODING.PARAMETERS.MATERIAL  = CN()
cfg.MODEL.DECODING.PARAMETERS.MATERIAL.include = True
cfg.MODEL.DECODING.PARAMETERS.MATERIAL.continuous = True
cfg.MODEL.DECODING.PARAMETERS.MATERIAL.num_category = -1