import os
from model.Train_Bert import training
# After Running prepocessing from the commandline we use the following

def run():
    training() # still need to distinguish between dev and trainfiles -> for loop for files

if __name__ == '__main__':
    run()