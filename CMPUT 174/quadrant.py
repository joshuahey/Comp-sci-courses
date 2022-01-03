#guessing game 
import random,string
import os
os.chdir(os.path.dirname(__file__))

#open instructions file

with open("instructionsgg.txt",'r') as ins:        
    print(ins.read())

total_guess_list=[]   

#playing the game per round
while 1:
    print("I am thinking of a letter between a and z")
    x=random.choice(string.ascii_letters).lower()  #the answer
    guess=''
    c,j=0,0
    k,v=[],[]   
    while guess!=x:
        c=c+1
        j=j+1
        guess=input("Take a guess:").lower()     #prompt for guess
        if guess.isalpha():
            if ord(guess) >ord(x):    #checking if guess higher than the answer
                print("Your guess is too high")
            elif ord(guess)<ord(x):   #checking if guess lower than the answer
                print("Your guess is too low")
            else:                     #guessed correctly
                print("Good job, you guessed the correct letter!")
            k.append(guess)           #appending the number of guesses each round
        else:
            print("Invalid")
    total_guess_list.append(j)         #appending the total number of guess list

    #my stats
    print("---MY STATS---")
    print("Number of guesses:",c)
    print("Level:expert" if c<5 else "Level:intermediate" if c in range(5,11) else "Level:beginner") #conditional statement to find level of the player

    #retrieving worst guess
    for i in k:
        v.append(abs(ord(i)-ord(x)))
    d=dict(zip(k,v))
    print("Worst Letter guess:",k[v.index(max(v))])
    #the below block of code is in case the dict function isnt accpeted
    # a=max(v)
    # for i in range(len(v)):
    #     if v[i]==a:
    #         print(k[i])
    # print(chr(v[-1]))
    
    #prompting user to continue or exit
    play_again=input("\nWould you like to play again? y/n")
    if play_again=="y":
        continue
    else:
        #get complete game statistics
        worst_guess=min(i for i in total_guess_list)      #worst guess
        best_guess=max(i for i in total_guess_list)      #best guess
        average=sum(i for i in total_guess_list)/len(total_guess_list) #average of all guesses
        print("Lowest Number of Guesses:",worst_guess)
        print("Highest Number of Guesses:",best_guess)
        print("Average Number of Guesses:",average)
        print("Overall Level:expert" if average<5 else "Overall Level:intermediate" if average in range(5,10) else "Overall Level:beginner")   #overall level
        break

