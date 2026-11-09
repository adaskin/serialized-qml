### Test accuracy (%), last epoch, mean ± std over seeds

| dataset | classes | T | majority class | logistic | MLP (L4-sized) | CNN (L4-sized) | grad. boosting | circuit L4 coherent, linear | circuit L4 measured, linear | amplitude 7q L4 | amplitude 7q L8 | circuit L4 coherent, MLP head | MLP sized to the MLP-head circuit | circuit L4 coherent, Z+ZZ+X+XX+Y+YY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| waveform | 3 | 6 | 33.9 ± 0.0 | 87.2 ± 0.1 | 87.1 ± 0.7 | - | 85.5 ± 0.4 | 85.0 ± 0.2 | 82.7 ± 2.5 | 86.2 ± 0.2 | 86.8 ± 0.5 | - | - | 86.0 ± 0.5 |
| wallrobot | 4 | 6 | 40.4 ± 0.0 | 68.3 ± 0.2 | 91.9 ± 0.3 | - | 99.9 ± 0.0 | 90.1 ± 0.6 | 87.0 ± 1.6 | 73.0 ± 1.1 | 77.1 ± 1.2 | - | - | 90.9 ± 1.2 |
| covertype | 7 | 14 | 48.8 ± 0.0 | 71.6 ± 0.1 | 75.8 ± 0.2 | - | 82.2 ± 0.1 | 73.8 ± 0.1 | 71.7 ± 0.4 | 71.7 ± 0.4 | 72.0 ± 0.2 | - | - | 73.2 ± 0.4 |
| letter | 26 | 4 | 4.1 ± 0.0 | 72.6 ± 0.2 | 80.4 ± 0.7 | - | 95.3 ± 0.1 | 71.0 ± 1.3 | 63.5 ± 2.6 | 59.7 ± 2.0 | 63.6 ± 0.6 | 85.6 ± 0.5 | 87.8 ± 0.2 | 79.9 ± 0.1 |
| fashion | 10 | 49 | 10.0 ± 0.0 | 82.6 ± 0.2 | 85.2 ± 0.1 | 85.7 ± 0.5 | 86.5 ± 0.3 | 82.7 ± 0.1 | - | 73.7 ± 0.4 | 74.4 ± 0.6 | - | - | - |
| bloodmnist | 8 | 49 | 19.5 ± 0.0 | 67.2 ± 0.3 | 70.3 ± 0.9 | 81.2 ± 0.3 | 86.0 ± 0.2 | 73.8 ± 1.1 | - | 58.0 ± 0.5 | 60.2 ± 0.2 | - | - | - |
| spambase | 2 | 15 | 60.5 ± 0.0 | 92.0 ± 0.1 | 93.4 ± 0.4 | - | 96.1 ± 0.2 | 91.7 ± 0.3 | 93.5 ± 0.5 | 90.0 ± 0.3 | 90.7 ± 0.7 | - | - | - |
| creditdefault | 2 | 6 | 77.9 ± 0.0 | 81.1 ± 0.1 | 81.7 ± 0.1 | - | 81.4 ± 0.2 | 81.9 ± 0.3 | 82.4 ± 0.1 | 79.6 ± 0.2 | 79.7 ± 0.0 | - | - | - |
| landsat | 6 | 9 | 23.9 ± 0.0 | 82.3 ± 0.1 | 89.2 ± 0.1 | - | 92.3 ± 0.4 | 89.3 ± 1.0 | 86.8 ± 0.8 | 78.7 ± 0.1 | 78.9 ± 0.2 | - | - | 89.7 ± 0.2 |
| drybean | 7 | 4 | 26.1 ± 0.0 | 92.1 ± 0.0 | 92.3 ± 0.1 | - | 92.8 ± 0.2 | 92.9 ± 0.1 | 92.8 ± 0.0 | 91.8 ± 0.1 | 91.8 ± 0.1 | - | - | 92.9 ± 0.1 |
| pneumonia | 2 | 49 | 62.5 ± 0.0 | 83.5 ± 0.1 | 84.1 ± 0.7 | 85.6 ± 0.7 | 84.9 ± 0.0 | 87.7 ± 0.6 | - | 81.2 ± 1.3 | 81.9 ± 1.0 | - | - | - |
| kmnist | 10 | 49 | 10.0 ± 0.0 | 68.2 ± 0.2 | 78.7 ± 0.1 | 80.3 ± 1.2 | 83.4 ± 0.2 | 70.4 ± 0.6 | - | 58.3 ± 0.6 | 62.6 ± 0.6 | - | - | - |

### Test balanced accuracy (%), last epoch, mean ± std over seeds

| dataset | classes | T | majority class | logistic | MLP (L4-sized) | CNN (L4-sized) | grad. boosting | circuit L4 coherent, linear | circuit L4 measured, linear | amplitude 7q L4 | amplitude 7q L8 | circuit L4 coherent, MLP head | MLP sized to the MLP-head circuit | circuit L4 coherent, Z+ZZ+X+XX+Y+YY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| waveform | 3 | 6 | 33.3 | 87.2 ± 0.1 | 87.1 ± 0.7 | - | 85.5 ± 0.4 | 85.0 ± 0.2 | 82.7 ± 2.5 | 86.2 ± 0.2 | 86.8 ± 0.5 | - | - | 86.0 ± 0.5 |
| wallrobot | 4 | 6 | 25.0 | 59.7 ± 0.8 | 90.8 ± 0.7 | - | 99.9 ± 0.0 | 88.8 ± 0.7 | 84.6 ± 2.2 | 66.2 ± 1.5 | 69.9 ± 0.8 | - | - | 89.6 ± 1.2 |
| covertype | 7 | 14 | 14.3 | 46.0 ± 0.9 | 56.2 ± 1.9 | - | 67.8 ± 0.1 | 45.3 ± 1.0 | 38.1 ± 2.1 | 42.3 ± 0.1 | 42.1 ± 1.4 | - | - | 49.2 ± 1.2 |
| letter | 26 | 4 | 3.8 | 72.5 ± 0.2 | 80.4 ± 0.7 | - | 95.3 ± 0.1 | 70.8 ± 1.3 | 63.5 ± 2.6 | 59.6 ± 2.0 | 63.5 ± 0.6 | 85.5 ± 0.5 | 87.7 ± 0.2 | 79.8 ± 0.1 |
| fashion | 10 | 49 | 10.0 | 82.6 ± 0.2 | 85.2 ± 0.1 | 85.7 ± 0.5 | 86.5 ± 0.3 | 82.7 ± 0.1 | - | 73.7 ± 0.4 | 74.4 ± 0.6 | - | - | - |
| bloodmnist | 8 | 49 | 12.5 | 67.2 ± 0.3 | 69.8 ± 0.9 | 79.5 ± 0.4 | 84.8 ± 0.2 | 73.0 ± 0.9 | - | 55.5 ± 0.4 | 57.4 ± 0.9 | - | - | - |
| spambase | 2 | 15 | 50.0 | 91.2 ± 0.1 | 92.9 ± 0.4 | - | 96.1 ± 0.2 | 91.4 ± 0.7 | 93.1 ± 0.6 | 89.2 ± 0.4 | 90.0 ± 0.7 | - | - | - |
| creditdefault | 2 | 6 | 50.0 | 61.2 ± 0.1 | 65.4 ± 0.2 | - | 65.0 ± 0.1 | 65.3 ± 0.6 | 66.3 ± 0.3 | 58.5 ± 0.3 | 59.1 ± 0.2 | - | - | - |
| landsat | 6 | 9 | 16.7 | 74.7 ± 0.3 | 86.5 ± 0.1 | - | 90.1 ± 0.5 | 86.8 ± 1.1 | 83.6 ± 1.1 | 69.3 ± 0.1 | 69.4 ± 0.2 | - | - | 87.1 ± 0.2 |
| drybean | 7 | 4 | 14.3 | 93.4 ± 0.0 | 93.5 ± 0.0 | - | 93.8 ± 0.2 | 93.9 ± 0.1 | 93.8 ± 0.1 | 92.9 ± 0.1 | 93.0 ± 0.1 | - | - | 94.0 ± 0.0 |
| pneumonia | 2 | 49 | 50.0 | 78.9 ± 0.1 | 79.3 ± 1.1 | 81.2 ± 1.0 | 80.8 ± 0.0 | 84.0 ± 0.8 | - | 76.1 ± 1.7 | 77.0 ± 1.5 | - | - | - |
| kmnist | 10 | 49 | 10.0 | 68.2 ± 0.2 | 78.7 ± 0.1 | 80.3 ± 1.2 | 83.4 ± 0.2 | 70.4 ± 0.6 | - | 58.3 ± 0.6 | 62.6 ± 0.6 | - | - | - |

### Test accuracy (%), best-validation checkpoint, mean ± std over seeds

| dataset | classes | T | majority class | logistic | MLP (L4-sized) | CNN (L4-sized) | grad. boosting | circuit L4 coherent, linear | circuit L4 measured, linear | amplitude 7q L4 | amplitude 7q L8 | circuit L4 coherent, MLP head | MLP sized to the MLP-head circuit | circuit L4 coherent, Z+ZZ+X+XX+Y+YY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| waveform | 3 | 6 | 33.9 ± 0.0 | 87.5 ± 0.4 | 86.3 ± 1.3 | - | 85.5 ± 0.4 | 85.2 ± 0.4 | 82.8 ± 2.6 | 86.2 ± 0.2 | 86.6 ± 0.2 | - | - | 86.1 ± 0.3 |
| wallrobot | 4 | 6 | 40.4 ± 0.0 | 68.4 ± 0.2 | 91.8 ± 0.2 | - | 99.9 ± 0.0 | 89.8 ± 0.3 | 87.0 ± 1.7 | 72.9 ± 1.2 | 76.2 ± 0.9 | - | - | 91.0 ± 1.6 |
| covertype | 7 | 14 | 48.8 ± 0.0 | 71.6 ± 0.1 | 75.9 ± 0.2 | - | 82.2 ± 0.1 | 73.8 ± 0.1 | 71.7 ± 0.5 | 71.8 ± 0.3 | 71.8 ± 0.2 | - | - | 73.3 ± 0.2 |
| letter | 26 | 4 | 4.1 ± 0.0 | 72.6 ± 0.2 | 80.4 ± 0.7 | - | 95.3 ± 0.1 | 71.0 ± 1.3 | 63.6 ± 2.6 | 59.5 ± 1.8 | 63.6 ± 0.6 | 85.5 ± 0.4 | 87.8 ± 0.2 | 79.8 ± 0.1 |
| fashion | 10 | 49 | 10.0 ± 0.0 | 82.7 ± 0.2 | 85.2 ± 0.1 | 85.7 ± 0.5 | 86.5 ± 0.3 | 82.8 ± 0.2 | - | 73.7 ± 0.4 | 74.3 ± 0.6 | - | - | - |
| bloodmnist | 8 | 49 | 19.5 ± 0.0 | 67.1 ± 0.4 | 70.4 ± 1.0 | 81.2 ± 0.3 | 86.0 ± 0.2 | 73.8 ± 1.0 | - | 57.8 ± 0.4 | 59.2 ± 1.2 | - | - | - |
| spambase | 2 | 15 | 60.5 ± 0.0 | 91.7 ± 0.3 | 93.6 ± 0.6 | - | 96.1 ± 0.2 | 90.7 ± 0.6 | 93.5 ± 0.6 | 89.8 ± 0.5 | 90.6 ± 0.6 | - | - | - |
| creditdefault | 2 | 6 | 77.9 ± 0.0 | 81.4 ± 0.2 | 81.7 ± 0.1 | - | 81.4 ± 0.2 | 82.1 ± 0.1 | 82.3 ± 0.2 | 79.1 ± 0.5 | 79.3 ± 0.7 | - | - | - |
| landsat | 6 | 9 | 23.9 ± 0.0 | 82.1 ± 0.3 | 89.2 ± 0.6 | - | 92.3 ± 0.4 | 89.2 ± 0.9 | 86.6 ± 0.9 | 78.9 ± 0.2 | 79.1 ± 0.3 | - | - | 89.5 ± 0.1 |
| drybean | 7 | 4 | 26.1 ± 0.0 | 92.0 ± 0.1 | 92.3 ± 0.4 | - | 92.8 ± 0.2 | 92.7 ± 0.3 | 92.7 ± 0.1 | 91.8 ± 0.1 | 91.8 ± 0.2 | - | - | 92.8 ± 0.1 |
| pneumonia | 2 | 49 | 62.5 ± 0.0 | 83.8 ± 0.3 | 82.6 ± 2.1 | 85.4 ± 0.8 | 84.9 ± 0.0 | 87.4 ± 1.6 | - | 81.2 ± 1.3 | 82.2 ± 0.8 | - | - | - |
| kmnist | 10 | 49 | 10.0 ± 0.0 | 68.0 ± 0.1 | 78.7 ± 0.2 | 80.1 ± 1.6 | 83.4 ± 0.2 | 70.6 ± 0.8 | - | 58.3 ± 0.6 | 62.5 ± 0.8 | - | - | - |

### Test balanced accuracy (%), best-validation checkpoint, mean ± std over seeds

| dataset | classes | T | majority class | logistic | MLP (L4-sized) | CNN (L4-sized) | grad. boosting | circuit L4 coherent, linear | circuit L4 measured, linear | amplitude 7q L4 | amplitude 7q L8 | circuit L4 coherent, MLP head | MLP sized to the MLP-head circuit | circuit L4 coherent, Z+ZZ+X+XX+Y+YY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| waveform | 3 | 6 | 33.3 | 87.5 ± 0.4 | 86.3 ± 1.3 | - | 85.5 ± 0.4 | 85.1 ± 0.4 | 82.9 ± 2.6 | 86.2 ± 0.2 | 86.6 ± 0.2 | - | - | 86.0 ± 0.3 |
| wallrobot | 4 | 6 | 25.0 | 59.8 ± 0.8 | 90.7 ± 0.7 | - | 99.9 ± 0.0 | 88.5 ± 0.3 | 84.4 ± 2.4 | 66.1 ± 1.7 | 68.9 ± 1.1 | - | - | 90.0 ± 1.3 |
| covertype | 7 | 14 | 14.3 | 45.8 ± 0.9 | 56.0 ± 1.8 | - | 67.8 ± 0.1 | 45.3 ± 1.0 | 38.1 ± 2.2 | 42.2 ± 0.3 | 41.5 ± 0.8 | - | - | 49.1 ± 2.0 |
| letter | 26 | 4 | 3.8 | 72.5 ± 0.2 | 80.3 ± 0.7 | - | 95.3 ± 0.1 | 70.8 ± 1.3 | 63.5 ± 2.6 | 59.4 ± 1.7 | 63.5 ± 0.6 | 85.4 ± 0.4 | 87.7 ± 0.2 | 79.7 ± 0.1 |
| fashion | 10 | 49 | 10.0 | 82.7 ± 0.2 | 85.2 ± 0.1 | 85.7 ± 0.5 | 86.5 ± 0.3 | 82.8 ± 0.2 | - | 73.7 ± 0.4 | 74.3 ± 0.6 | - | - | - |
| bloodmnist | 8 | 49 | 12.5 | 67.1 ± 0.5 | 69.9 ± 0.9 | 79.4 ± 0.5 | 84.8 ± 0.2 | 72.7 ± 1.1 | - | 55.7 ± 0.5 | 57.0 ± 1.2 | - | - | - |
| spambase | 2 | 15 | 50.0 | 90.9 ± 0.3 | 93.1 ± 0.5 | - | 96.1 ± 0.2 | 89.7 ± 0.9 | 93.2 ± 0.7 | 89.0 ± 0.6 | 89.8 ± 0.6 | - | - | - |
| creditdefault | 2 | 6 | 50.0 | 62.5 ± 0.9 | 65.4 ± 0.3 | - | 65.0 ± 0.1 | 64.9 ± 0.2 | 66.3 ± 0.6 | 56.7 ± 2.4 | 57.1 ± 3.3 | - | - | - |
| landsat | 6 | 9 | 16.7 | 74.4 ± 0.4 | 86.6 ± 1.0 | - | 90.1 ± 0.5 | 87.0 ± 0.9 | 83.5 ± 1.1 | 69.6 ± 0.3 | 69.9 ± 0.4 | - | - | 86.9 ± 0.3 |
| drybean | 7 | 4 | 14.3 | 93.1 ± 0.3 | 93.6 ± 0.3 | - | 93.8 ± 0.2 | 93.6 ± 0.3 | 93.8 ± 0.2 | 92.9 ± 0.1 | 93.0 ± 0.2 | - | - | 93.8 ± 0.1 |
| pneumonia | 2 | 49 | 50.0 | 79.3 ± 0.4 | 77.3 ± 3.0 | 80.9 ± 1.1 | 80.8 ± 0.0 | 83.6 ± 2.2 | - | 76.1 ± 1.7 | 77.3 ± 1.3 | - | - | - |
| kmnist | 10 | 49 | 10.0 | 68.0 ± 0.1 | 78.7 ± 0.2 | 80.1 ± 1.6 | 83.4 ± 0.2 | 70.6 ± 0.8 | - | 58.3 ± 0.6 | 62.5 ± 0.8 | - | - | - |

### Circuit runs (100 epochs unless named _e10)

| run | params | train (last) | val (last) | test (last) | test (best val) | best-val epoch | balanced test (last) | hours |
|---|---|---|---|---|---|---|---|---|
| bloodmnist_coherent_L4_linear_seed42 | 7839 | 75.6 | 73.7 | 72.4 | 72.4 | 90 | 72.0 | 12.65 |
| bloodmnist_coherent_L4_linear_seed42_e10 | 7839 | 67.3 | 66.4 | 66.4 | 66.4 | 10 | 66.5 | 1.83 |
| bloodmnist_coherent_L4_linear_seed43 | 7839 | 77.6 | 75.8 | 74.9 | 74.9 | 90 | 74.2 | 12.67 |
| bloodmnist_coherent_L4_linear_seed44 | 7839 | 76.8 | 75.8 | 74.2 | 74.2 | 100 | 72.9 | 13.23 |
| covertype_coherent_L4_linear_seed42 | 2364 | 75.2 | 72.9 | 73.9 | 73.9 | 100 | 46.1 | 5.08 |
| covertype_coherent_L4_linear_seed42_e10 | 2364 | 67.9 | 66.9 | 68.0 | 68.0 | 10 | 33.1 | 0.38 |
| covertype_coherent_L4_linear_seed42_rozzzxxxyyy | 2560 | 74.6 | 71.1 | 72.6 | 73.1 | 90 | 49.3 | 9.13 |
| covertype_coherent_L4_linear_seed43 | 2364 | 75.4 | 73.3 | 73.9 | 73.9 | 100 | 45.9 | 4.87 |
| covertype_coherent_L4_linear_seed43_rozzzxxxyyy | 2560 | 75.8 | 72.9 | 73.5 | 73.2 | 80 | 47.7 | 9.43 |
| covertype_coherent_L4_linear_seed44 | 2364 | 73.9 | 72.1 | 73.6 | 73.6 | 100 | 43.9 | 4.36 |
| covertype_coherent_L4_linear_seed44_rozzzxxxyyy | 2560 | 74.6 | 72.5 | 73.5 | 73.5 | 100 | 50.7 | 7.44 |
| covertype_measured_L4_linear_seed42 | 2364 | 72.4 | 71.1 | 72.2 | 72.4 | 90 | 39.8 | 20.95 |
| covertype_measured_L4_linear_seed43 | 2364 | 72.3 | 70.3 | 71.7 | 71.7 | 100 | 39.3 | 22.95 |
| covertype_measured_L4_linear_seed44 | 2364 | 70.6 | 68.6 | 71.2 | 71.0 | 90 | 35.1 | 18.81 |
| creditdefault_coherent_L4_linear_seed42 | 1041 | 82.9 | 81.6 | 82.0 | 82.1 | 20 | 65.4 | 2.01 |
| creditdefault_coherent_L4_linear_seed43 | 1041 | 82.7 | 81.4 | 82.2 | 82.2 | 10 | 66.0 | 2.19 |
| creditdefault_coherent_L4_linear_seed44 | 1041 | 83.4 | 81.2 | 81.5 | 82.0 | 20 | 64.6 | 2.06 |
| creditdefault_measured_L4_linear_seed42 | 1041 | 82.4 | 82.1 | 82.3 | 82.3 | 70 | 65.9 | 11.59 |
| creditdefault_measured_L4_linear_seed43 | 1041 | 82.2 | 82.2 | 82.5 | 82.6 | 90 | 66.6 | 8.54 |
| creditdefault_measured_L4_linear_seed44 | 1041 | 82.4 | 82.1 | 82.4 | 82.1 | 20 | 66.4 | 5.44 |
| drybean_coherent_L4_linear_seed42 | 804 | 93.2 | 92.3 | 92.8 | 92.6 | 50 | 93.7 | 1.21 |
| drybean_coherent_L4_linear_seed42_rozzzxxxyyy | 1000 | 93.4 | 92.7 | 92.9 | 92.7 | 40 | 94.0 | 1.48 |
| drybean_coherent_L4_linear_seed43 | 804 | 92.9 | 92.8 | 92.8 | 92.3 | 50 | 93.9 | 1.24 |
| drybean_coherent_L4_linear_seed43_rozzzxxxyyy | 1000 | 93.1 | 93.6 | 92.8 | 92.8 | 80 | 94.0 | 1.40 |
| drybean_coherent_L4_linear_seed44 | 804 | 93.2 | 93.2 | 93.1 | 93.1 | 80 | 94.0 | 1.24 |
| drybean_coherent_L4_linear_seed44_rozzzxxxyyy | 1000 | 93.2 | 93.8 | 93.1 | 92.9 | 80 | 94.0 | 1.29 |
| drybean_measured_L4_linear_seed42 | 804 | 93.0 | 92.4 | 92.8 | 92.8 | 100 | 93.9 | 7.43 |
| drybean_measured_L4_linear_seed43 | 804 | 92.7 | 92.7 | 92.8 | 92.8 | 100 | 93.9 | 4.68 |
| drybean_measured_L4_linear_seed44 | 804 | 92.5 | 93.2 | 92.7 | 92.6 | 40 | 93.7 | 3.44 |
| fashion_coherent_L4_linear_seed42 | 7869 | 85.5 | 84.0 | 82.6 | 82.6 | 100 | 82.6 | 15.23 |
| fashion_coherent_L4_linear_seed42_e10 | 7869 | 76.2 | 77.1 | 75.7 | 75.7 | 10 | 75.7 | 2.19 |
| fashion_coherent_L4_linear_seed43 | 7869 | 85.7 | 83.2 | 82.8 | 82.8 | 100 | 82.9 | 14.62 |
| fashion_coherent_L4_linear_seed44 | 7869 | 87.3 | 84.5 | 82.8 | 82.9 | 90 | 82.8 | 14.82 |
| kmnist_coherent_L4_linear_seed42 | 7869 | 87.1 | 82.6 | 70.0 | 70.0 | 100 | 70.0 | 27.84 |
| kmnist_coherent_L4_linear_seed43 | 7869 | 87.6 | 84.6 | 69.9 | 70.2 | 90 | 69.9 | 27.09 |
| kmnist_coherent_L4_linear_seed44 | 7869 | 87.4 | 83.6 | 71.2 | 71.7 | 90 | 71.2 | 28.10 |
| landsat_coherent_L4_linear_seed42 | 1569 | 89.3 | 88.1 | 88.2 | 88.2 | 100 | 85.8 | 2.69 |
| landsat_coherent_L4_linear_seed42_rozzzxxxyyy | 1737 | 91.3 | 89.9 | 89.7 | 89.4 | 90 | 87.1 | 2.77 |
| landsat_coherent_L4_linear_seed43 | 1569 | 90.2 | 87.7 | 89.0 | 89.1 | 70 | 86.2 | 2.62 |
| landsat_coherent_L4_linear_seed43_rozzzxxxyyy | 1737 | 91.1 | 89.5 | 89.4 | 89.4 | 100 | 86.9 | 2.91 |
| landsat_coherent_L4_linear_seed44 | 1569 | 90.8 | 90.7 | 90.6 | 90.4 | 80 | 88.3 | 2.45 |
| landsat_coherent_L4_linear_seed44_rozzzxxxyyy | 1737 | 90.9 | 90.7 | 89.8 | 89.7 | 50 | 87.4 | 2.62 |
| landsat_measured_L4_linear_seed42 | 1569 | 88.1 | 87.4 | 87.8 | 87.8 | 100 | 85.0 | 7.98 |
| landsat_measured_L4_linear_seed43 | 1569 | 86.5 | 85.8 | 86.6 | 86.0 | 70 | 83.4 | 7.93 |
| landsat_measured_L4_linear_seed44 | 1569 | 85.8 | 86.2 | 85.9 | 85.9 | 80 | 82.3 | 4.25 |
| letter_coherent_L4_linear_seed42 | 1089 | 69.8 | 68.3 | 69.1 | 69.1 | 100 | 69.0 | 0.66 |
| letter_coherent_L4_linear_seed42_e10 | 1089 | 49.1 | 46.1 | 49.4 | 49.4 | 10 | 49.1 | 0.06 |
| letter_coherent_L4_linear_seed42_rozzzxxxyyy | 1817 | 82.3 | 77.9 | 79.8 | 79.8 | 100 | 79.8 | 0.75 |
| letter_coherent_L4_linear_seed43 | 1089 | 72.8 | 73.2 | 72.0 | 72.0 | 100 | 71.8 | 0.90 |
| letter_coherent_L4_linear_seed43_rozzzxxxyyy | 1817 | 81.7 | 81.2 | 79.7 | 79.7 | 100 | 79.6 | 0.71 |
| letter_coherent_L4_linear_seed44 | 1089 | 72.8 | 72.4 | 71.8 | 71.8 | 100 | 71.7 | 0.81 |
| letter_coherent_L4_linear_seed44_rozzzxxxyyy | 1817 | 82.0 | 79.9 | 80.0 | 79.9 | 90 | 80.0 | 0.70 |
| letter_coherent_L4_mlp_seed42 | 3349 | 90.5 | 84.4 | 85.4 | 85.3 | 90 | 85.4 | 0.57 |
| letter_coherent_L4_mlp_seed43 | 3349 | 89.5 | 85.4 | 86.3 | 86.0 | 90 | 86.2 | 0.66 |
| letter_coherent_L4_mlp_seed44 | 3349 | 88.7 | 86.2 | 85.0 | 85.0 | 100 | 85.0 | 0.67 |
| letter_coherent_L9_linear_seed42 | 1869 | 71.4 | 68.3 | 70.1 | 70.1 | 100 | 69.9 | 1.11 |
| letter_coherent_L9_linear_seed43 | 1869 | 74.0 | 75.4 | 74.1 | 74.1 | 100 | 74.0 | 1.10 |
| letter_coherent_L9_linear_seed44 | 1869 | 73.4 | 71.9 | 71.4 | 71.5 | 90 | 71.3 | 1.10 |
| letter_measured_L4_linear_seed42 | 1089 | 68.6 | 65.8 | 67.1 | 67.2 | 80 | 67.1 | 3.07 |
| letter_measured_L4_linear_seed43 | 1089 | 61.2 | 63.8 | 61.4 | 61.4 | 100 | 61.3 | 11.87 |
| letter_measured_L4_linear_seed44 | 1089 | 62.5 | 63.1 | 62.1 | 62.1 | 100 | 62.0 | 6.54 |
| pneumonia_coherent_L4_linear_seed42 | 7749 | 99.0 | 96.4 | 87.2 | 87.2 | 100 | 83.3 | 16.41 |
| pneumonia_coherent_L4_linear_seed43 | 7749 | 98.8 | 95.8 | 88.5 | 89.6 | 60 | 85.1 | 16.13 |
| pneumonia_coherent_L4_linear_seed44 | 7749 | 98.7 | 96.0 | 87.3 | 85.6 | 70 | 83.5 | 15.19 |
| spambase_coherent_L4_linear_seed42 | 2445 | 95.9 | 91.3 | 91.3 | 90.8 | 40 | 90.5 | 3.13 |
| spambase_coherent_L4_linear_seed43 | 2445 | 96.1 | 93.2 | 92.1 | 89.9 | 20 | 92.1 | 3.21 |
| spambase_coherent_L4_linear_seed44 | 2445 | 96.4 | 91.8 | 91.8 | 91.3 | 90 | 91.6 | 2.92 |
| spambase_measured_L4_linear_seed42 | 2445 | 95.0 | 94.0 | 92.8 | 92.6 | 90 | 92.3 | 9.70 |
| spambase_measured_L4_linear_seed43 | 2445 | 95.0 | 96.5 | 93.8 | 93.8 | 100 | 93.5 | 9.53 |
| spambase_measured_L4_linear_seed44 | 2445 | 95.8 | 93.5 | 93.8 | 94.0 | 50 | 93.5 | 6.01 |
| wallrobot_coherent_L4_linear_seed42 | 1071 | 92.1 | 89.9 | 89.5 | 89.5 | 90 | 88.3 | 0.35 |
| wallrobot_coherent_L4_linear_seed42_e10 | 1071 | 76.3 | 78.4 | 74.5 | 74.5 | 10 | 69.5 | 0.04 |
| wallrobot_coherent_L4_linear_seed42_rozzzxxxyyy | 1183 | 93.3 | 88.8 | 92.2 | 92.8 | 90 | 91.0 | 0.42 |
| wallrobot_coherent_L4_linear_seed43 | 1071 | 90.4 | 87.6 | 90.1 | 89.6 | 80 | 88.3 | 0.60 |
| wallrobot_coherent_L4_linear_seed43_rozzzxxxyyy | 1183 | 93.1 | 85.8 | 89.3 | 88.8 | 70 | 88.0 | 0.42 |
| wallrobot_coherent_L4_linear_seed44 | 1071 | 90.8 | 88.8 | 90.8 | 90.2 | 70 | 89.7 | 0.47 |
| wallrobot_coherent_L4_linear_seed44_rozzzxxxyyy | 1183 | 92.3 | 87.8 | 91.2 | 91.5 | 90 | 89.9 | 0.42 |
| wallrobot_measured_L4_linear_seed42 | 1071 | 88.5 | 85.8 | 89.4 | 89.4 | 100 | 87.5 | 2.56 |
| wallrobot_measured_L4_linear_seed43 | 1071 | 86.6 | 86.0 | 85.8 | 85.7 | 80 | 82.2 | 1.74 |
| wallrobot_measured_L4_linear_seed44 | 1071 | 86.2 | 85.1 | 86.0 | 85.8 | 90 | 84.1 | 4.28 |
| waveform_coherent_L4_linear_seed42 | 1056 | 87.6 | 85.8 | 85.1 | 85.1 | 100 | 85.1 | 0.33 |
| waveform_coherent_L4_linear_seed42_e10 | 1056 | 81.6 | 81.2 | 80.1 | 80.1 | 10 | 80.1 | 0.04 |
| waveform_coherent_L4_linear_seed42_rozzzxxxyyy | 1140 | 87.5 | 84.8 | 86.7 | 86.3 | 70 | 86.7 | 0.40 |
| waveform_coherent_L4_linear_seed43 | 1056 | 86.9 | 86.3 | 84.7 | 84.7 | 100 | 84.7 | 0.45 |
| waveform_coherent_L4_linear_seed43_rozzzxxxyyy | 1140 | 87.3 | 86.0 | 85.5 | 85.6 | 70 | 85.5 | 0.39 |
| waveform_coherent_L4_linear_seed44 | 1056 | 87.8 | 86.3 | 85.1 | 85.7 | 90 | 85.1 | 0.47 |
| waveform_coherent_L4_linear_seed44_rozzzxxxyyy | 1140 | 88.1 | 88.2 | 85.9 | 86.3 | 60 | 85.9 | 0.39 |
| waveform_measured_L4_linear_seed42 | 1056 | 81.1 | 78.0 | 79.6 | 79.8 | 70 | 79.6 | 2.37 |
| waveform_measured_L4_linear_seed43 | 1056 | 85.7 | 88.0 | 85.8 | 86.1 | 80 | 85.8 | 1.62 |
| waveform_measured_L4_linear_seed44 | 1056 | 82.9 | 84.8 | 82.7 | 82.7 | 100 | 82.7 | 4.30 |

### Sizes and parameter counts

| dataset | n_train | n_val | n_test | circuit L4 linear | amplitude 7q L4 | amplitude 7q L8 | circuit L4 MLP head | MLP (L4-sized) |
|---|---|---|---|---|---|---|---|---|
| waveform | 3601 | 400 | 999 | 1056 | 201 | 357 | - | 1053 |
| wallrobot | 3929 | 436 | 1091 | 1071 | 216 | 372 | - | 1077 |
| covertype | 10000 | 2000 | 9999 | 2364 | 261 | 417 | - | 2363 |
| letter | 10000 | 1600 | 4000 | 1089 | 546 | 702 | 3349 | 1101 |
| fashion | 10000 | 2000 | 10000 | 7869 | 306 | 462 | - | 7876 |
| bloodmnist | 8779 | 1527 | 3421 | 7839 | 276 | 432 | - | 7798 |
| spambase | 3313 | 368 | 920 | 2445 | 186 | 342 | - | 2462 |
| creditdefault | 10000 | 2000 | 6000 | 1041 | 186 | 342 | - | 1042 |
| landsat | 4634 | 514 | 1287 | 1569 | 246 | 402 | - | 1554 |
| drybean | 9801 | 1089 | 2721 | 804 | 261 | 417 | - | 799 |
| pneumonia | 4708 | 524 | 624 | 7749 | 186 | 342 | - | 7763 |
| kmnist | 10000 | 2000 | 10000 | 7869 | 306 | 462 | - | 7876 |
