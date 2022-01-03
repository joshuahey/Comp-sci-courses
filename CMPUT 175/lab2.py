import os
os.chdir(os.path.dirname(__file__))

def getInputfile():
    x=input("Enter file name:")
    while x.endswith(".txt") is False:
        x=input("Try again:")  
    return x

def decrypt(filename):
    """[summary]

    Args:
        filename ([type]): [description]
    """
    with open(filename,"r") as f:
        a=f.readlines()   
        m = int(a[0])
        for i in a:
            b=i.strip().split()
        for i in b:
            for j in i:
                z=ord(j.lower())
                z=(z-m)%26
                print(chr(z),end="")
            print(" ",end="")

decrypt(getInputfile())

