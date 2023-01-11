import pytest


import sys
 
# setting path
sys.path.append('../src')
 
# importing
from src.model import Encoder,Decoder,Model
 


#Test if test works
def test_test():
    assert True

#Test if function can be calles
def test_init():
    enc = Encoder(10,10,10)
    dec = Decoder(10,10,10)
    model = Model(enc,dec)
    assert model != None, "Chech if model is created"

#Test if function can be calles
def test_init():
    enc = Encoder(10,10,10)
    dec = Decoder(10,10,10)
    model = Model(enc,dec)
    for idx, m in enumerate(enc.named_modules()):
        assert m != None, "Chech if model is created"
        
