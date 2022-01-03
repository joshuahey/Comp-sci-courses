#--------------------------------------------------------------------
# Stack implementation #2 
# (Top of stack corresponds to back of list)
# 
# Author: CMPUT 175 team
# References: CMPUT 175 lecture slides (Stacks), except for __str__
# #--------------------------------------------------------------------

# class Stack:
#     def __init__(self):
#         self.items = []
    
#     def push(self, item):
#         self.items.append(item)
    
#     def pop(self):
#         return self.items.pop()
    
#     def peek(self):
#         return self.items[len(self.items)-1] 
    
#     def isEmpty(self):
#         return self.items == []
    
#     def size(self):
#         return len(self.items)
    
#     def show(self):
#         print(self.items)
    
#     def __str__(self):
#         stackAsString = 'bottom -> '
#         for item in self.items:
#             stackAsString += item + ' '
#         stackAsString += '<- top'
#         return stackAsString

# musicIdol=Stack()
# musicIdol.push("Drake")
# musicIdol.push("Beyonce")
# musicIdol.push("Bieber")
# musicIdol.push("Drake")
# print(musicIdol.size())

import time
start=time.time()
def iseven(n):
    if n==0:
        return True
    elif n==1:
        return False
    else:
        return iseven(n-2)

end=time.time()
n=int(input("Hi"))
print(iseven(n))
print(end-start)