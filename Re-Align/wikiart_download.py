import openxlab
openxlab.login(ak="gw3ozvaxpzkkxr50dbra", sk="dpaqznjlvee2jrom5ker11nj3w9nvamzb3bgkkq1")

from openxlab.dataset import get
get(dataset_repo='OpenDataLab/WikiArt', target_path='/data2/gaodz/WikiArt')