from .. import model_factory
from .dewarpnet import LitDewarpNetBM, LitDewarpNetJoint, LitDewarpNetWC

model_factory.register_model("dewarpnet_wc", LitDewarpNetWC)
model_factory.register_model("dewarpnet_bm", LitDewarpNetBM)
model_factory.register_model("dewarpnet_joint", LitDewarpNetJoint)
