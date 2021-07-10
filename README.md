# cmtf EEG-fMRI
Coupled matrix-tensor factorization for integrating EEG and fMRI in the source space  

I proposed a novel algorithm to integrate two modalities that contain complementary information on brain function:
* EEG measures brain potentials in milliseconds temporal resolution from scalp however EEG signals are affected from volume conduction which requires source localization to infer the real neural generators in the brain.
* fMRI measures metabolic signal called BOLD that is related to neural activity in a high spatial resolution of millimeters however its temporal resolution is limited.

Integration of these brain signals that are recorded simulatneously can give more information on brain function.

The algorithm presented here simulatenously decomposes EEG and fMRI signals by imposing several constraints on the spatial and temporal components.

Technical explanation and derivations can be found in the pdf file. 

Reference: Karahan et al, Proc IEEE, 2015, https://ieeexplore.ieee.org/abstract/document/7214360
