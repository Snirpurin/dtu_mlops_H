import pytest
import vae_mnist.py
import model.py

#Test if test works
def test_test():
    assert True

#Test if function can be calles
def test_init():
    enc = Encoder(10,10,10)
    dec = Decoder(10,10,10)
    model = Model(end,dec)
    assert model != None

