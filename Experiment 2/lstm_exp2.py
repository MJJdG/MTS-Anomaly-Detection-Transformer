#!/usr/bin/env python
# coding: utf-8
"""
Initializing and running LSTM models for the anomaly prediction experiment
@author: Max de Grauw (M.degrauw@student.ru.nl)
"""

from MTS_lstm import mtsLSTM

if __name__ == '__main__':
    _ = mtsLSTM('m1', 'synthetic_L1_300_L2_30_SIG_20_SEED_48', 300, 30, 21, 4, 512)
    _ = mtsLSTM('m2', 'synthetic_L1_300_L2_90_SIG_20_SEED_48', 300, 90, 21, 4, 512)
    _ = mtsLSTM('m3', 'synthetic_L1_300_L2_150_SIG_20_SEED_48', 300, 150, 21, 4, 512)
    _ = mtsLSTM('m4', 'synthetic_L1_600_L2_60_SIG_20_SEED_48', 600, 60, 21, 4, 1024)
    _ = mtsLSTM('m5', 'synthetic_L1_600_L2_180_SIG_20_SEED_48', 600, 180, 21, 4, 1024)
    _ = mtsLSTM('m6', 'synthetic_L1_600_L2_300_SIG_20_SEED_48', 600, 300, 21, 4, 1024)   
    _ = mtsLSTM('m7', 'synthetic_L1_300_L2_30_SIG_40_SEED_48', 300, 30, 41, 6, 512)
    _ = mtsLSTM('m8', 'synthetic_L1_300_L2_90_SIG_40_SEED_48', 300, 90, 41, 6, 512)
    _ = mtsLSTM('m9', 'synthetic_L1_300_L2_150_SIG_40_SEED_48', 300, 150, 41, 6, 512)
    _ = mtsLSTM('m10', 'synthetic_L1_600_L2_60_SIG_40_SEED_48', 600, 60, 41, 6, 512)
    _ = mtsLSTM('m11', 'synthetic_L1_600_L2_180_SIG_40_SEED_48', 600, 180, 41, 6, 512)
    _ = mtsLSTM('m12', 'synthetic_L1_600_L2_300_SIG_40_SEED_48', 600, 300, 41, 6, 512)