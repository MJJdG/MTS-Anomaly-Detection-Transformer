# -*- coding: utf-8 -*-
"""
Data generation for the anomaly prediction experiment
@author: Max de Grauw (M.degrauw@student.ru.nl)
"""

import multiprocessing as mp
from MTS_datagenerator import generationPipeline
        
if __name__ == '__main__': # L2 = 10%, 30%, 50%
    processes = [mp.Process(target=generationPipeline, args=(300, 30, "synthetic", 48, 1, 20, 2**14, 2**12, 2**12, 2**13)),
                 mp.Process(target=generationPipeline, args=(300, 90, "synthetic", 48, 1, 20, 2**14, 2**12, 2**12, 2**13)),
                 mp.Process(target=generationPipeline, args=(300, 150, "synthetic", 48, 1, 20, 2**14, 2**12, 2**12, 2**13)),
                 mp.Process(target=generationPipeline, args=(600, 60, "synthetic", 48, 1, 20, 2**14, 2**12, 2**12, 2**13)),
                 mp.Process(target=generationPipeline, args=(600, 180, "synthetic", 48, 1, 20, 2**14, 2**12, 2**12, 2**13)),
                 mp.Process(target=generationPipeline, args=(600, 300, "synthetic", 48, 1, 20, 2**14, 2**12, 2**12, 2**13)),
                 mp.Process(target=generationPipeline, args=(300, 30, "synthetic", 48, 1, 40, 2**14, 2**12, 2**12, 2**13)),
                 mp.Process(target=generationPipeline, args=(300, 90, "synthetic", 48, 1, 40, 2**14, 2**12, 2**12, 2**13)),
                 mp.Process(target=generationPipeline, args=(300, 150, "synthetic", 48, 1, 40, 2**14, 2**12, 2**12, 2**13)),
                 mp.Process(target=generationPipeline, args=(600, 60, "synthetic", 48, 1, 40, 2**14, 2**12, 2**12, 2**13)),
                 mp.Process(target=generationPipeline, args=(600, 180, "synthetic", 48, 1, 40, 2**14, 2**12, 2**12, 2**13)),
                 mp.Process(target=generationPipeline, args=(600, 300, "synthetic", 48, 1, 40, 2**14, 2**12, 2**12, 2**13))]
    for p in processes:
        p.start() # Start processes in parallel
    for p in processes:
        p.join() # Wait for all processes to finish
