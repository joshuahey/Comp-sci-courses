try:
    print("Raising an exception")
    raise Exception('CMPUT', '175')
except Exception as inst: # the exception instance
    print(inst.args) # arguments stored in .args
    x, y = inst.args # unpack args
    print('x =', x,'y =', y)
