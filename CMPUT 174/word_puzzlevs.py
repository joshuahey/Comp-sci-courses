import random
import os
os.chdir(os.path.dirname(__file__))


#function to guess the answer
def guess(answer): 
    print(answer)
    guess_list=[]
    for i in range(len(answer)):
        guess_list.append("_")    #creating a list to store guesses
    n=4
    print("The answer so far is"," ".join(guess_list))
    while n>0:
        m=input()         #prompting user for input/guess
        if m in answer:   #checking if input/guess character is in the answer/word to be guessed
            for j in range(len(answer)):
                if answer[j] == m:   #if position of character in answer is the guessed character
                    guess_list[j]=answer[j]  #replace the character in guess_list in respective positions
            print("Guess a letter(",n,"guesses remaining):")
            print("The answer so far is"," ".join(guess_list))
        elif m not in answer:
            n-=1
            print("Guess a letter(",n,"guesses remaining):")
            print("The answer so far is"," ".join(guess_list))
        if n==0:
            print("Not quite, the correct word was",answer,".Better luck next time")
            continue
        elif "".join(guess_list)==answer:
            print("Good job! You found the word",answer)
            break

#main function
def main():
    with open('wp_instructions.txt','r') as  w:
        print(w.read())
        w.close()
    a=['apple', 'banana', 'watermelon', 'kiwi', 'pineapple', 'mango']
    for i in range(len(a)):
        answer=random.choice(a)
    guess(answer)
    input("Press enter to end the game.")
main()

