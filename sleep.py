
import parser
import time
import argparse

# from CLI
parser = argparse.ArgumentParser()
parser.add_argument("--time", type=int, default=10, help="Number of minutes for sleep")
parser.add_argument("--unit", type=str, default='h', help="Unit of time for sleep")

args = parser.parse_args()

n_time = args.time
unit = args.unit
if unit == 'h':
    n_time *= 3600
elif unit == 'm':
    n_time *= 60
elif unit == 's':
    n_time *= 1
else:
    raise ValueError("Unit not recognized")

print("=> sleeping for {} seconds".format(n_time))
time.sleep(n_time)