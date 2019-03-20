
import subprocess
import os

import numpy as np


PROGRAM_CMD = 'python'
PYTHON_FILE = 'policy-tune.py'
LOG_FILE = 'policylog'

ARG_DICT = {"--no-cuda": "",
            "--gamma": 0,
            "--fg-size": 0,
            "--batch-size": 0,
            "--max-iterations": 0,
            "--learning-rate": 0,
            "--epsilon": 0
            }


GAMMA = [0.99] # np.linspace(0.8, 1, num = 5, endpoint = True)
FG_SIZE = [100] # np.arange(0, 100 + 1, 10)
BATCH_SIZE = [32]
MAX_ITR = [250] #np.arange(2000, 10000 + 1, 1000)
LR = np.linspace(0.0001, 0.01, num = 5, endpoint = True)
EPSILON = [1.0] # np.linspace(0.1, 1.0, num = 2, endpoint = True)


def createCmd(args, output):
    cmd = " ".join( [PROGRAM_CMD, PYTHON_FILE] +
                        [ key + " " + str(value) for key, value in args.items()] )

    cmd = cmd #  + " > " + output
    # print(cmd)
    return cmd


def createFilename(args):
    name = LOG_FILE + "".join([ key + str(value) for key, value in args.items()] ) + ".txt"
    # print(name)
    return name





if __name__ == '__main__':
    for gamma in GAMMA:
        for fg in FG_SIZE:
            for batch in BATCH_SIZE:
                for itr in MAX_ITR:
                    for lr in LR:
                        for eps in EPSILON:
                            args = {"--no-cuda": "",
                                    "--gamma": gamma,
                                    "--fg-size": fg,
                                    "--batch-size": batch,
                                    "--max-iterations": itr,
                                    "--learning-rate": lr,
                                    "--epsilon": eps
                                    }

                            # create filename without --log-file argument
                            output = createFilename(args)

                            # create command to run
                            cmd = createCmd(args, output)
                            print(cmd)
                            subprocess.call(cmd, shell = True)
