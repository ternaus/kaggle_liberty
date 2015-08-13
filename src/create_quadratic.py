from __future__ import division

__author__ = 'Vladimir Iglovikov'

import graphlab as gl
from graphlab.toolkits.feature_engineering import *

joined = gl.SFrame('../data/joined.csv')

features = [
  # 'Hazard',
  # 'Id',
  'T1_V1', 'T1_V10', 'T1_V13', 'T1_V14', 'T1_V17', 'T1_V2', 'T1_V3', 'T1_V6', 'T2_V1', 'T2_V10',
  'T2_V11', 'T2_V12', 'T2_V14', 'T2_V15', 'T2_V2', 'T2_V3', 'T2_V4', 'T2_V6', 'T2_V7', 'T2_V8', 'T2_V9', 'tp_0', 'tp_1', 'tp_2', 'tp_3', 'tp_4', 'tp_5', 'tp_6', 'tp_7', 'tp_8', 'tp_9', 'tp_10', 'tp_11', 'tp_12', 'tp_13', 'tp_14', 'tp_15', 'tp_16', 'tp_17', 'tp_18', 'tp_19', 'tp_20', 'tp_21', 'tp_22', 'tp_23', 'tp_24', 'tp_25', 'tp_26', 'tp_27', 'tp_28', 'tp_29', 'tp_30', 'tp_31', 'tp_32', 'tp_33', 'tp_34', 'tp_35', 'tp_36', 'tp_37', 'tp_38', 'tp_39', 'tp_40', 'tp_41', 'tp_42', 'tp_43', 'tp_44', 'tp_45', 'tp_46', 'tp_47', 'tp_48', 'tp_49', 'tp_50', 'tp_51', 'tp_52', 'tp_53', 'tp_54', 'tp_55', 'tp_56', 'tp_57', 'tp_58', 'tp_59', 'tp_60', 'tp_61', 'tp_62', 'tp_63', 'tp_64', 'tp_65', 'tp_66', 'tp_67', 'tp_68', 'tp_69', 'tp_70', 'tp_71', 'tp_72', 'tp_73', 'tp_74', 'tp_75', 'tp_76', 'tp_77', 'tp_78', 'tp_79', 'tp_80', 'tp_81', 'tp_82', 'tp_83', 'tp_84']


quadratic = gl.feature_engineering.create(joined, QuadraticFeatures(features=features))

joined_new = quadratic.transform(joined)
joined_new.save('../data/joined_q.csv')