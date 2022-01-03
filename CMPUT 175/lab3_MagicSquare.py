#----------------------------------------------------
# Lab 3: Magic Square class
# 
# Author: 
# Collaborators/References:
#----------------------------------------------------

class MagicSquare:
    def __init__(self, n):
        '''
        Initializes an empty square with n*n cells.
        Inputs:  
           n (int) - number of rows in square, or equivalently, the number of columns
        Returns: None
        '''       
        self.square = []  # list of lists, where each internal list represents a row
        self.size = n  # number of columns and rows of square
        
        for i in range(self.size):
            row = []
            for j in range(self.size):
                row.append(0)
            self.square.append(row)
            
                
    def cellIsEmpty(self, row, col):
        '''
        Checks if a given cell is empty, or if it already contains a number 
        greater than 0.
        Inputs:
           row (int) - row index of cell to check
           col (int) - column index of cell to check
        Returns: True if cell is empty; False otherwise
        '''
        return self.square[row][col] == 0

    
    def drawSquare(self):
        '''
        Displays the current state of the square, formatted with column and row 
        indices shown along the top and left side, respectively.
        Inputs: N/A
        Returns: None
        '''        
        for k in range(0,self.size):
            print("{:4d}".format(k),end='   ')
        print("")
        for i in range(0,self.size):
            print("  +"+"-"*6*(self.size)+"+") 
            print(i,end=' ')
            for j in range(0,self.size):
                if self.square[i][j]==0:
                    print("|","{:4s}".format("."),end='')
                else:
                    print("|","{:<4d}".format(self.square[i][j]),end='')
            print("|")
        print("  +"+"-"*6*self.size+"+")


    def update(self, row, col, num):
        '''
        Assigns the integer, num, to the cell at the provided row and column, 
        but only if that cell is empty and only if the number isn't already 
        in another cell in the square (i.e. it is unique)
        Inputs:
           row (int) - row index of cell to update
           col (int) - column index of cell to update
           num (int) - entry to place in cell
        Returns: True if attempted update was successful; False otherwise
        '''       
        # check uniqueness
        unique = True
        for valuesInRow in self.square:
            if num in valuesInRow:
                unique = False
        
        # try to update cell
        if unique:
            if row < self.size and row >= 0:
                if col < self.size and col >= 0:
                    if self.cellIsEmpty(row, col):
                        self.square[row][col] = num
                        return True
        return False
    
    
    def isFull(self):
        '''
        Checks if the square has any remaining empty cells.
        Inputs: N/A
        Returns: True if the square has no empty cells (full); False otherwise
        '''
        full = False
        for row in self.square:
            if 0 in row:
                full = False
                return full
            return True
        
           
    def isMagic(self):
        '''
        Checks whether the square is a complete, magic square:
          1. All cells contain a positive number (i.e. greater than 0)
          2. All lines sum up to the same value (horizontals, verticals, diagonals)
          
        Inputs: N/A
        Returns: True if square is magic; False otherwise
        '''          
        b={}      
        a=set(b)
        sumdiagonal1=0
        sumdiagonal2=0
        for i in range(self.size):
            sumdiagonal1+=self.square[i][i]
            sumdiagonal2+=self.square[i][self.size-i-1]
        a.add(sumdiagonal1)
        a.add(sumdiagonal2)
        for i in range(self.size):
            sumrow=0
            sumcolumn=0
            for j in range(self.size):
                sumrow+=self.square[i][j]
                sumcolumn+=self.square[j][i]
            a.add(sumrow)
            a.add(sumcolumn)
        for i in range(self.size):
            for j in range(self.size):
                if len(a)!=1 and self.square[i][i]>0:
                    return False
                return True
     

if __name__ == "__main__":
    # TEST EACH METHOD THOROUGHLY HERE
    # complete the suggested tests; more tests may be required
    
    # start by creating an empty 3x3 square and checking the contents of the square attribute
    mySquare = MagicSquare(3)
    # print(mySquare.square)

    # check if a specific cell (any cell) is empty, as expected.
    # for i in range(0,3):
    #     for j in range(0,3):
    #         print(mySquare.cellIsEmpty(i,j))
    # does the entire square display properly when you draw it?
    # mySquare.drawSquare()

    # assign a number to an empty cell and display the entire square
    # mySquare.update(0,0,10)
    # mySquare.drawSquare()
    
    # try to assign a number to a non-empty cell. What happens?

    # mySquare.update(0,0,19)
    # mySquare.drawSquare()
    # doesnt replace the current element

    # check if the square is full. Should it be full after only 1 entry?

    # print(mySquare.isFull())
    # no

    
    # check if the square is a magic square. Should it be after only 1 entry?

    # print(mySquare.isMagic())
    # after 1 entry no as it wont account for the other rows and column sums
    
    # add values to the square so that every line adds up to 21. Display the square.
    #(Check out the example at the beginning of the lab description for values to put into each cell.)
    mySquare.update(0,0,10)
    mySquare.update(0,1,3)
    mySquare.update(0,2,8)
    mySquare.update(1,0,5)
    mySquare.update(1,1,7)
    mySquare.update(1,2,9)
    mySquare.update(2,0,6)
    mySquare.update(2,1,11)
    mySquare.update(2,2,4)
    mySquare.drawSquare()
    
    # # check if the square is full
    print(mySquare.isFull())
    
    # # check if the square is magic
    print(mySquare.isMagic())
    
    # write additional tests, as needed
    

    
