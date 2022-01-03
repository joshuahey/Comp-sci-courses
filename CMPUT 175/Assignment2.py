# ASSIGNMENT 2
# AUTHORS: JOSHUA GEORGE
# COLLABORATORS: NONE



import random, os
# the reason i import os is because my laptop cant read the files 
os.chdir(os.path.dirname(__file__))

class Card:

    def __init__(self,rank,suit):
        self.rank=rank
        self.suit=suit
        self.visible=0

    def __str__(self):
        return f"{self.rank}{self.suit} " if self.visible else "?? " 
        
    def __repr__(self):
        visible = "+" if self.visible else "-"
        return f"{self.rank}{self.suit}{visible} "

    def isvisible(self):
        return self.visible==1

    def faceupcard(self, visible):
        self.visible=visible

    def rankcard(self):
        return self.rank

    def suitcard(self):
        return self.suit


class Deck:
    def __init__(self,name):
        self.__name=name
        self.__cards=[]

    def __str__(self):
        outString = "[ "
        for i in range(len(self.__cards)-1,-1,-1):
            outString += str(self.__cards[i])
        outString += "]"
        return outString

    def __repr__(self):
        outString = "[ "
        for i in range(len(self.__cards)-1,-1,-1):
            outString += repr(self.__cards[i])
        outString += "]"
        return outString

    def namedeck(self):
        return self.__name

    def sizedeck(self):
        return len(self.__cards)

    def isemptydeck(self):
        return self.__cards==[]

    def pushdeck(self,item):
        return self.__cards.append(item)

    def popdeck(self):
        return self.__cards.pop()

    def peekdeck(self):
        return self.__cards[len(self.__cards)-1] 
    
# create objects
Stock=Deck("Stock")
Discard=Deck("Discard")
Spades=Deck("Spades")
Hearts=Deck("Hearts")
Diamonds=Deck("Diamonds")
Clubs=Deck("Clubs")
Pile1=Deck("PILE-1")
Pile2=Deck("PILE-2")
Pile3=Deck("PILE-3")
Pile4=Deck("PILE-4")
Pile5=Deck("PILE-5")
Pile6=Deck("PILE-6")
Pile7=Deck("PILE-7")


piles = [Pile1, Pile2, Pile3, Pile4, Pile5, Pile6, Pile7]
piles_names=[Pile1.namedeck(), Pile2.namedeck(), Pile3.namedeck(), Pile4.namedeck(), Pile5.namedeck(), Pile6.namedeck(), Pile7.namedeck()]
suit=[Spades, Hearts, Diamonds, Clubs]
suit_names=[Spades.namedeck(), Hearts.namedeck(), Diamonds.namedeck(), Clubs.namedeck()]
total=piles+suit
total.append(Stock)
total_={}
# for mapping say Pile-x --> x
for i in range(len(total)):
    total_[total[i]]=str(int(i+1))

# for mapping ranks say A --> 0
ranks=["A","2","3","4","5","6","7","8","9","T","J","Q","K"]
order={}
for i in range(len(ranks)):
    order[ranks[i]]=i

# for mapping ranks say suit--> suitcard
suits_=["d","s","c","h"]
correspond={Spades:"s",Hearts:"h",Diamonds:"d",Clubs:"c"}

# this is only for when i create a new game ie load2 function from the program now when i read from a file
# ranks=random.sample(ranks,len(ranks))

def load(file_name):
    """[load function (from a file version)]
    Args:
        file_name ([file]): [file]
    """
    for line in file_name:        
        stripped_line = line.strip()
        line_list = stripped_line.split()
        # to input all the card objects in deck objects
        for i in range(len(line_list)-2,1,-1):
            # check for format error
            try:
                if line_list[0]=="Stock":
                    # create card onject
                    x=Card(line_list[i][0],line_list[i][1])
                    # append to deck
                    Stock.pushdeck(x)
                    # check for visibility
                    if line_list[i][2] == "+":
                        x.faceupcard(True)
                    elif line_list[i][2] == "-":
                        x.faceupcard(False)
                elif line_list[0]=="Discard":
                    x=Card(line_list[i][0],line_list[i][1])
                    Discard.pushdeck(x)
                    if line_list[i][2] == "+":
                        x.faceupcard(True)
                    elif line_list[i][2] == "-":
                        x.faceupcard(False)
                elif line_list[0] in suit_names:
                    for j in range(len(suit)):
                        if suit_names[j] == line_list[0]:
                            x=Card(line_list[i][0],line_list[i][1])
                            suit[j].pushdeck(x)
                            if line_list[i][2] == "+":
                                x.faceupcard(True)
                            elif line_list[i][2] == "-":
                                x.faceupcard(False)
                elif line_list[0] in piles_names:
                    for j in range(len(piles_names)):
                        if piles_names[j] == line_list[0]:
                            x=Card(line_list[i][0],line_list[i][1])
                            piles[j].pushdeck(x)
                            if line_list[i][2] == "+":
                                x.faceupcard(True)
                            elif line_list[i][2] == "-":
                                x.faceupcard(False)
            except IndexError:
                print("format error for %s in line %s" %(repr(x),stripped_line))
                        
def save(file_name):
    """[save to a file]
    write to a file
    Args:
        file_name ([file]): [file]
    """
    total_2=[]
    total_2.append(Stock)
    total_2.append(Discard)
    total_2+=suit
    total_2+=piles
    for i in total_2:
        file_name.write(i.namedeck())
        file_name.write(" ")
        file_name.write(repr(i))
        file_name.write("\n")

    # i was confused about the save function so i did this so that whenever i save i clear the board otherwise 
    for i in total_2:
        for j in range(i.sizedeck()):
            i.popdeck()
    

def load2():
    # this function is for creating a new game from scratch (no reading files here)
    for i in suits_:
        for j in ranks:
            Stock.pushdeck(Card(j,i))
    for i in range(len(piles)):
        for j in range(i+1):
            piles[i].pushdeck(Stock.popdeck())
        piles[i].peekdeck().faceupcard(True)


def cheat():
    """[prints visible board]
    reprshows visibility
    """
    print("Klondike!")
    # prints stock n discard
    if Stock.isemptydeck()!=True:
        Stock.peekdeck().faceupcard(True)
    print(Stock.namedeck(),"  ",repr(Stock))
    print(Discard.namedeck(),"",repr(Discard))
    # prints suits
    for i in range(len(suit)):
        if suit[i].isemptydeck()== True:
            suit[i]==[]
            print("{:<8s} {}".format(suit[i].namedeck(),repr(suit[i])))
        else:
            suit[i].peekdeck().faceupcard(True)
            print("{:<8s} {}".format(suit[i].namedeck(),repr(suit[i])))
     # prints piles
    for i in range(len(piles)):
        if piles[i].isemptydeck() == True:
            piles[i]==[] 
        else:
            piles[i].peekdeck().faceupcard(True)
        print(piles[i].namedeck()," ",repr(piles[i]))


def board():
    """[prints board hidden]
    str doesnt shows visibility
    """
    print("Klondike!")
    # prints stock n discard
    if Stock.isemptydeck()!=True:
        Stock.peekdeck().faceupcard(True)
    print(Stock.namedeck(),"  ",str(Stock))
    print(Discard.namedeck(),"",str(Discard))
    # prints suits
    for i in range(len(suit)):
        if suit[i].isemptydeck()== True:
            suit[i]==[]
            print("{:<8s} {}".format(suit[i].namedeck(),str(suit[i])))
        else:
            suit[i].peekdeck().faceupcard(True)
            print("{:<8s} {}".format(suit[i].namedeck(),str(suit[i])))
    # prints piles
    for i in range(len(piles)):
        if piles[i].isemptydeck() == True:
            piles[i]==[] 
        else:       
            piles[i].peekdeck().faceupcard(True)
        print(piles[i].namedeck()," ",str(piles[i]))



def move(first,second):
    """[move function]
    My logic for move function is just popping/iterating and popping and then pushing to the second deck by comparing the ranks and in some cases 
    suits with the first deck and second deck
    Args:
        first ([list]): [list]
        second ([list]): [list]
    """
    # check popping from empty deck using assert
    assert first.sizedeck() > 0, ("%s empty" %(first.namedeck()))

    # piles to piles
    # my logic for moving many cards from one pile to another is by just iterating over all faceup cards and checking their value 
    # suppose we have a move 8d 9c 10h to another pile 10c this will move 8d and 9c and show an invalid message for 10h as it checks for that as well
    # the invalid message shows for the normal cases ie say 5c --->Jh or 10h-->Ac

    if second in piles:
        dummy=[]
        for i in range(first.sizedeck()-1,-1,-1):
            if first.peekdeck().isvisible() == True:
                dummy.append(first.popdeck())
        dummy.reverse()
        for i in dummy:
            if second.isemptydeck()!=True:
                if order[second.peekdeck().rankcard()]-1 == order[i.rankcard()] :
                    second.pushdeck(i)    
                else:
                    print("Invalid move for {}".format(i))
                    first.pushdeck(i)
            else:
                # if second pile is empty only king can move
                if second.isemptydeck()==True and i.rankcard()=="K":
                    second.pushdeck(i)    
                else:
                    print("Invalid move for {}".format(i))
                    first.pushdeck(i)

        if first.isemptydeck() == True:
            first=[]
        else:
            first.peekdeck().faceupcard(True) 
        

    # piles to suits
    elif  second in suit:
        if second.isemptydeck()!=True:
            if order[first.peekdeck().rankcard()]-1 == order[second.peekdeck().rankcard()] :
                second.pushdeck(first.popdeck())
            else:
                print("Invalid move for {}".format(first.peekdeck()))
        # if suit is empty only A can go to a suit
        elif second.isemptydeck()==True:
            if first.peekdeck().rankcard() == "A":
                second.pushdeck(first.popdeck())
            else:
                first.pushdeck(first.popdeck())
                print("Invalid move for  {}".format(first.peekdeck()))
        
        if first.isemptydeck() == True:
            first=[]
        else:
            first.peekdeck().faceupcard(True)

    # stock to piles and suits
    elif second in total:
        if second.isemptydeck()!=True:
            if order[first.peekdeck().rankcard()]==order[second.peekdeck().rankcard()]-1 and second in piles:
                second.pushdeck(first.popdeck())
            elif order[first.peekdeck().rankcard()]-1==order[second.peekdeck().rankcard()] and second in suit :
                second.pushdeck(first.popdeck())
            else:
                print("Invalid move for {}".format(first.peekdeck()))
        else:
        # if second pile is empty only King can be pushd and if second suit is empty only A can be pushed
            if first.peekdeck().rankcard()=="K" and second in piles:
                second.pushdeck(first.popdeck())
            elif first.peekdeck().rankcard()=="A" and second in suit :
                second.pushdeck(first.popdeck())
            else:
                print("Invalid move for {}".format(first.peekdeck()))
        # and first.peekdeck().suitcard()
        if first.isemptydeck() == True:
            first=[]
        else:
            first.peekdeck().faceupcard(True) 



def discard():
    """[moves <=3 number of cards from stock and move thems to discard deck]
    """
    try:
        if Stock.sizedeck()==1:
            Stock.peekdeck().faceupcard(False)
            Discard.pushdeck(Stock.popdeck())
        elif Stock.sizedeck()==2:
            for i in range(0,2):
                Stock.peekdeck().faceupcard(False)
                Discard.pushdeck(Stock.popdeck()) 
        else:
            for i in range(0,3):
                Stock.peekdeck().faceupcard(False)
                Discard.pushdeck(Stock.popdeck())
    except IndexError:
        print("%s is empty"%(Stock.namedeck()))

       
def reset():
    """[move all cards from discard to stock deck]
    """
    # here i used a dummy list (list2) to store the objects and reverse it and push it to the stock in the required order
    if Stock.isemptydeck()==True:
        list2=[]
        for i in range(Discard.sizedeck()):
            list2.append(Discard.popdeck())
        list2.reverse()
        for i in list2:
            Stock.pushdeck(i)


def comment(Executing):
    """[return the move as a string]
    Args:
        Executing ([list]): [list]
    Returns:
        [string]: [string rep of the move]
    """
    x=" ".join(Executing)
    x=x.title()
    return x
    

def done():
    """[exit the program if done is inputted]
    """
    print("Thank you for playing")
    exit()


def main():
    """[main function]
    """
    play=True
    print("Welcome to Klondike!")
    
    while play:
        # win condition
        if Spades.sizedeck()== Diamonds.sizedeck()==Hearts.sizedeck()==Clubs.sizedeck() ==13:
            print("Congratulations!!")

        input_=input("Your move:")
        suitable_input=["board","move","1","2","3","4","5","6","suit","Stock","Discard","load"]
        Executing = input_.split()

        if Executing[0]=="load":
            try:
                with open(Executing[1],"r+") as file:
                    load(file)
                    print("Executing:",Executing)
            except OSError:
                print("%s does not exist"%(Executing[1]))
                

        elif Executing[0]=="save":
            try:
                with open(Executing[1],"r+") as file:
                    save(file)
                    print("Executing:",Executing) 
            except OSError:
                print("%s does not exist"%(Executing[1]) )
                

        elif Executing[0]=="load2":
            print("Executing:",Executing)
            load2()

        elif Executing[0]=="board":
            print("Executing:",Executing)
            board()

        elif Executing[0]=="cheat":
            print("Executing:",Executing)
            cheat()

        elif Executing[0]=="move":
        
            print("Executing:",Executing)
            for i in total:
                for j in total:
                    if Executing[1] == total_[i] and Executing[2] == total_[j]:
                        move(i,j)
            for i in total:
                if Executing[1] == total_[i] and Executing[2] == "suit": 
                    if i.peekdeck().suitcard() == correspond[Spades]:
                        move(i,Spades) 
                    elif i.peekdeck().suitcard() == correspond[Hearts]:
                        move(i,Hearts) 
                    elif i.peekdeck().suitcard() == correspond[Diamonds]:
                        move(i,Diamonds) 
                    elif i.peekdeck().suitcard() == correspond[Clubs]:
                        move(i,Clubs)   
            for i in total:
                if Executing[1] == "stock" and Executing[2] == total_[i]:
                    move(Stock,i)
            if Executing[1] == "stock" and Executing[2] == "suit":
                if Stock.peekdeck().suitcard() == correspond[Spades]:
                    move(Stock,Spades) 
                elif Stock.peekdeck().suitcard() == correspond[Hearts]:
                    move(Stock,Hearts) 
                elif Stock.peekdeck().suitcard() == correspond[Diamonds]:
                    move(Stock,Diamonds) 
                elif Stock.peekdeck().suitcard() == correspond[Clubs]:
                    move(Stock,Clubs)        

        elif Executing[0]=="done":
            print("Executing:",Executing)
            done()

        elif Executing[0]=="discard":
            print("Executing:",Executing)
            discard()

        elif Executing[0]=="reset":
            print("Executing:",Executing)
            reset()

        elif Executing[0]=="comment":
            print("Executing:",Executing)
            print(comment(Executing))
               
main()


