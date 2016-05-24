# Example use: cat log.txt | python parse_kld_from_log.py
# writes to kld_matrix.pkl

import sys
from collections import OrderedDict
import numpy
import cPickle

epochs = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("module kld --"):
        continue
    start = line.find("--") + 3
    fields = [x.strip().split(':') for x in line[start:].split(',') if len(x)]
    fields = sorted(fields, key=lambda x: x[0])
    fields = OrderedDict(map(lambda (x, y): (x, float(y)), fields))
    epochs.append(fields)

assert len(epochs) > 0
headers = epochs[0].keys()

statistics = numpy.zeros((len(epochs), len(headers)), dtype="float32")
print "Found headers: %s" % headers
for num, epoch in enumerate(epochs):
    statistics[num] = [epoch.get(x) for x in headers]
cPickle.dump(statistics, open("kld_matrix.pkl", "w"))
