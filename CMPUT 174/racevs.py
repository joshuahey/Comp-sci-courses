# Racing game first to cross the line wins
import random
# die roll
def roll_die():
    t=random.randint(1,6)
    return t
# state the condition of the game
def state_game(b,a):
    print("*"*80)
    print("Player X"," ".join(b))
    print("Player Y"," ".join(a))
    print("*"*80)
# update the position for player x 
# b=x_list
# g=die throw number
# j=keeping track of previous position in order to get swapped with g+j in the next throw 
def update_game_x(b,g,j):
    if g+j>12:     #if die throws sum exceed range of list/line
        for h in range(len(b)):
            b[h],b[0]="-","x"
        print("The roll was too high, player X has been sent to the start")         
    else:
        b[(g+j)],b[j]=b[j],b[(g+j)]  #the swapping
    return b,j
# update the position for player o 
# a=x_list
# k=die throw number
# l=keeping track of previous position in order to get swapped with g+j in the next throw 
def update_game_o(a,k,l):
    if k+l>12:   #if die throws sum exceed range of list/line
        for u in range(len(a)):
            a[u],a[0]="-","o"
        print("The roll was too high, player O has been sent to the start")
    else:        
        a[(k+l)],a[l]=a[l],a[(k+l)]  
    return a,l
#check to see who won
def check_result(b,a):
    if b[12]=="x":
        print("Player X has won!")
    elif a[12]=="o":
        print("Player O has won!")
#main function calling other function
def main():
    x_list,o_list=[],[]   #lists for both the players
    x_list = ["-" for i in range(13)]
    o_list = ["-" for i in range(13)]
    print("Players begin in the starting position")
    x_list[0],o_list[0]="x","o"
    f=True #bool condition for alternate plays
    j,l=0,0
    while x_list[12]!="x" and o_list[12]!="o":
        state_game(x_list,o_list)
        if f==True :
            input("Player X press enter to roll")
            g=roll_die()
            print("Player X rolled a ",g)
            update_game_x(x_list,g,j)
            j+=g
            if j>12:   #if sum is greater than range of list then reset the game
                j=0
            f=False
            check_result(x_list,o_list)
        else:
            input("Player O press enter to roll")
            k=roll_die()
            print("Player O rolled a ",k)
            update_game_o(o_list,k,l)
            l+=k
            if l>12:    #if sum is greater than range of list then reset the game
                l=0
            f=True
            check_result(x_list,o_list)

main()
