import numpy as np
results = [l.split(' ') for l in open('RESULTS.txt').readlines()]
nlls = []
for line in results:
    for i in range(len(line)):
        if line[i].strip('\n').find('vfe/iwae') > -1:
            nlls.append(float(line[i+1].strip('\n')))
nlls = np.asarray(nlls)
smoothed_nlls = []
for i in range(nlls.shape[0]):
    if i > 25:
        v = np.mean(nlls[i-25:i])
        smoothed_nlls.append(v)
smoothed_nlls = np.asarray(smoothed_nlls)