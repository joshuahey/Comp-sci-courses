import os
os.chdir(os.path.dirname(__file__))

with open("xx.txt","r") as o:
    print(o.read())