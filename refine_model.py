import torch
import argparse
parser = argparse.ArgumentParser('parameters')
parser.add_argument('--exp-name', type=str, default='oct28', help="exp name")
args = parser.parse_args()
state_dict = torch.load("runs/%s/%s.pt" % (args.exp_name, args.exp_name), map_location=torch.device("cuda:1"))
torch.save(state_dict, "runs/%s/%s.pt" % (args.exp_name, args.exp_name), _use_new_zipfile_serialization=False)