import numpy as np
import sys
import os


if len(sys.argv) > 1:
  TEST_DIR = sys.argv[1]
else:
  raise RuntimeError('No test directory provided')
GT_DIR = 'labeled'

def get_mse(gt, test):
  test = np.nan_to_num(test)
  return np.mean(np.nanmean((gt - test)**2, axis=0))


zero_mses = []
mses = []

for i in range(0,5):
  gt = np.loadtxt(os.path.join(GT_DIR, f'{i}.txt'))
  zero_mses.append(get_mse(gt, np.zeros_like(gt)))

  test = np.loadtxt(os.path.join(TEST_DIR, f'{i}.txt'))
  mses.append(get_mse(gt, test))

percent_err_vs_all_zeros = 100*np.mean(mses)/np.mean(zero_mses)
print(f'YOUR ERROR SCORE IS {percent_err_vs_all_zeros:.2f}% (lower is better)')
