import os
import model.Train_Bert
# After Running prepocessing from the commandline we use the following

def run():
   model.Train_Bert.training() # still need to distinguish between dev and trainfiles -> for loop for files

if __name__ == '__main__':
    run()