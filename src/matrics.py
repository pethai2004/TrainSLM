
import torch   

class Metrics:
    
    '''Class to compute and report relevant metrics for the training process'''
    
    def __init__(self, writer=None, speed_matric: bool = True):
        self.speed_matric = speed_matric  
        self.writer = writer
        self.matric = {}
        
    def report_matric(self):
        pass 