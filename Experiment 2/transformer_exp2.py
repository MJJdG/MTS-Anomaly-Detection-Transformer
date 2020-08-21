#!/usr/bin/env python
# coding: utf-8
"""
Initializing and running Transformer models for the anomaly prediction experiment
@author: Max de Grauw (M.degrauw@student.ru.nl)
"""

from MTS_transformer import mtsTransformerForecasting, mtsTransformerPrediction

if __name__ == '__main__':
    anompred, anompred_ffw = mtsTransformerForecasting('m1a', 'synthetic_L1_300_L2_30_SIG_20_SEED_48', 300, 30,\
                                                        21, 1, 1024, 32, batch_size=16, epochs=300, learning_rate =1e-4)
    _, _ = mtsTransformerPrediction('m1b', anompred, anompred_ffw, 'synthetic_L1_300_L2_30_SIG_20_SEED_48', 300, 30,\
                              21, 1, 1024, 32, batch_size=16, epochs=300, learning_rate =1e-4, loss_weight=1, folds=4)

    anompred, anompred_ffw = mtsTransformerForecasting('m2a', 'synthetic_L1_300_L2_90_SIG_20_SEED_48', 300, 30,\
                                                        21, 1, 1024, 32, batch_size=16, epochs=300, learning_rate =1e-4)
    _, _ = mtsTransformerPrediction('m2b', anompred, anompred_ffw, 'synthetic_L1_300_L2_90_SIG_20_SEED_48', 300, 30,\
                              21, 1, 1024, 32, batch_size=16, epochs=300, learning_rate =1e-4, loss_weight=1, folds=4)
                              
    anompred, anompred_ffw = mtsTransformerForecasting('m3a', 'synthetic_L1_300_L2_150_SIG_20_SEED_48', 300, 30,\
                                                        21, 1, 1024, 32, batch_size=16, epochs=300, learning_rate =1e-4)
    _, _ = mtsTransformerPrediction('m3b', anompred, anompred_ffw, 'synthetic_L1_300_L2_150_SIG_20_SEED_48', 300, 30,\
                              21, 1, 1024, 32, batch_size=16, epochs=300, learning_rate =1e-4, loss_weight=1, folds=4)

    anompred, anompred_ffw = mtsTransformerForecasting('m4a', 'synthetic_L1_600_L2_60_SIG_20_SEED_48', 300, 30,\
                                                        21, 2, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4)
    _, _ = mtsTransformerPrediction('m4b', anompred, anompred_ffw, 'synthetic_L1_600_L2_60_SIG_20_SEED_48', 300, 30,\
                              21, 2, 1024, 32, batch_size=16, epochs=300, learning_rate =1e-4, loss_weight=1, folds=4) 
                              
    anompred, anompred_ffw = mtsTransformerForecasting('m5a', 'synthetic_L1_600_L2_180_SIG_20_SEED_48', 300, 30,\
                                                        21, 2, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4)
    _, _ = mtsTransformerPrediction('m5b', anompred, anompred_ffw, 'synthetic_L1_600_L2_180_SIG_20_SEED_48', 300, 30,\
                              21, 2, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4, loss_weight=1, folds=4)

    anompred, anompred_ffw = mtsTransformerForecasting('m6a', 'synthetic_L1_600_L2_300_SIG_20_SEED_48', 300, 30,\
                                                        21, 2, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4)
    _, _ = mtsTransformerPrediction('m6b', anompred, anompred_ffw, 'synthetic_L1_600_L2_300_SIG_20_SEED_48', 300, 30,\
                              21, 2, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4, loss_weight=1, folds=4)
                              
    anompred, anompred_ffw = mtsTransformerForecasting('m7a', 'synthetic_L1_300_L2_30_SIG_40_SEED_48', 300, 30,\
                                                        21, 1, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4)
    _, _ = mtsTransformerPrediction('m7b', anompred, anompred_ffw, 'synthetic_L1_300_L2_30_SIG_40_SEED_48', 300, 30,\
                              21, 1, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4, loss_weight=1, folds=4)

    anompred, anompred_ffw = mtsTransformerForecasting('m8a', 'synthetic_L1_300_L2_90_SIG_40_SEED_48', 300, 30,\
                                                        21, 1, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4)
    _, _ = mtsTransformerPrediction('m8b', anompred, anompred_ffw, 'synthetic_L1_300_L2_90_SIG_40_SEED_48', 300, 30,\
                              21, 1, 1024, 32, batch_size=16, epochs=300, learning_rate =1e-4, loss_weight=1, folds=4) 
                              
    anompred, anompred_ffw = mtsTransformerForecasting('m9a', 'synthetic_L1_300_L2_150_SIG_40_SEED_48', 300, 30,\
                                                        21, 1, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4)
    _, _ = mtsTransformerPrediction('m9b', anompred, anompred_ffw, 'synthetic_L1_300_L2_150_SIG_40_SEED_48', 300, 30,\
                              21, 1, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4, loss_weight=1, folds=4)

    anompred, anompred_ffw = mtsTransformerForecasting('m10a', 'synthetic_L1_600_L2_60_SIG_40_SEED_48', 300, 30,\
                                                        21, 1, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4)
    _, _ = mtsTransformerPrediction('m10b', anompred, anompred_ffw, 'synthetic_L1_600_L2_60_SIG_40_SEED_48', 300, 30,\
                              21, 1, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4, loss_weight=1, folds=4)
                              
    anompred, anompred_ffw = mtsTransformerForecasting('m11a', 'synthetic_L1_600_L2_180_SIG_40_SEED_48', 300, 30,\
                                                        21, 1, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4)
    _, _ = mtsTransformerPrediction('m11b', anompred, anompred_ffw, 'synthetic_L1_600_L2_180_SIG_40_SEED_48', 300, 30,\
                              21, 1, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4, loss_weight=1, folds=4)

    anompred, anompred_ffw = mtsTransformerForecasting('m12a', 'synthetic_L1_600_L2_300_SIG_40_SEED_48', 300, 30,\
                                                        21, 1, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4)
    _, _ = mtsTransformerPrediction('m12b', anompred, anompred_ffw, 'synthetic_L1_600_L2_300_SIG_40_SEED_48', 300, 30,\
                              21, 1, 1024, 16, batch_size=16, epochs=300, learning_rate =1e-4, loss_weight=1, folds=4)                               
