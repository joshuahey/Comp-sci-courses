#----------------------------------------------------
# Lab 4: Web browser simulator
# Purpose of code:
#
# Author: 
# Collaborators/references:
#----------------------------------------------------

from stack import Stack

def getAction():
    """[check action of  user]
    """
    user_input= input("Enter = to enter a URL, < to go back, > to go forward, q to quit:")
    z=['=','<','>','q']
    if user_input not in z:
        print("Invalid entry")
    return user_input

def goToNewSite(current, bck, fwd):
    """[go to the site entered]
    Args:
        current ([string]): [website address]
        bck ([stack]): [back stack]
        fwd ([stack]): [forward stack]
    """
    bck.push(current)
    if not fwd.isEmpty():
        for i in range(fwd.size()):
            fwd.pop()
    current=input("URL:")
    print(current)
    return current


def goBack(current, bck, fwd):
    """[go to previous website]
    Args:
        current ([string]): [website adress]
        bck ([stack]): [back stack]
        fwd ([stack]): [forward stack]
    """
    if bck.isEmpty():
        print("Cannot go back")
    else:
        fwd.push(current)
        current=bck.pop()
    return current

def goForward(current, bck, fwd):
    """[summary]
    Args:
        current ([string]): [website address]
        bck ([stack]): [back stack]
        fwd ([stack]): [forward stack]
    """
    if fwd.isEmpty():
        print("Cannot go forward")  
    else:
        bck.push(current)
        current=fwd.pop()
    return current

def main():
    HOME = 'www.cs.ualberta.ca'
    back = Stack()
    forward = Stack()
    
    
    current = HOME
    quit = False
    
    while not quit:
        print('\nCurrently viewing', current)
        action = getAction()
        
        if action == '=':
            current = goToNewSite(current, back, forward)
        elif action == '<':
            current = goBack(current, back, forward)
        elif action == '>':
            current = goForward(current, back, forward)
        elif action == 'q':
            quit = True
    
    print('Browser closing...goodbye.')    

        
if __name__ == "__main__":
    main()
    