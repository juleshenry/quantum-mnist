import os
from pathlib import Path
plank_dir = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/zooplankton_0p5x'
plank_dir = (os.listdir(plank_dir))
plank=sorted(filter(lambda x:x[0]!='.', plank_dir))
cartesian_names = []
cartesian_names = [(a,b,) for a in plank for b in plank if a!=b]
print((cartesian_names))
assert len(cartesian_names) + len(plank) == len(plank)**2
print(cartesian_names[0])
import tensorflow