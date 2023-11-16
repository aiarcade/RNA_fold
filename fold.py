from arnie.bpps import bpps


bpps_dict = {}
my_sequence = 'CGCUGUCUGUACUUGUAUCAGUACACUGACGAGUCCCUAAAGGACGAAACAGCG'

for pkg in ['eternafold']:
    bpps_dict[pkg] = bpps(my_sequence, package=pkg)
print(bpps_dict)