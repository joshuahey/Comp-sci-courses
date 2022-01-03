#program showing overlay of a map via cells
import os
os.chdir(os.path.dirname(__file__))

def create_grid(filename):
    """[Description: Create a nested list based on the data given in a file]
    list_1=representing rows
    list_2=representing columns
    Args:
        filename: [A string representing the name of a file]
    Returns:
        [list]: [A two dimensional nested list populated with data]
    """
    d_3=filename.read().splitlines()
    list_1=[]
    m=0
    for i in range(0,int(d_3[0])):
        list_2=[]
        for j in range(m,int(d_3[1])+m):
            list_2.append(d_3[j+2])
            m+=1
        list_1.append(list_2)
    return list_1

def display_grid(grid):
    """[Display the grid as seen in the screenshot.]
    Args:
        grid ([list]): [ A two dimensional nested list]
    """
    for i in range(0,len(grid)):
        for j in range(0,len(grid[0])):
            print("|",grid[i][j],end = " ")
        print("|")

def find_neighbors(row_index,col_index, grid):
    """[Find all the neighbors of a particular cell in the grid]
    list_3=representing the neighbors of a given cell
    Args:
        row_index ([int]): [An int representing the row index]
        col_index ([int]): [An int representing the column index]
        grid ([list]): [A two dimensional nested list]
    Returns:
        [list]: [A list with all the neighbors of a given cell]
    """
    list_3=[] 
    for i in range(row_index - 1, row_index + 2):
        for j in range(col_index - 1, col_index +2):
            if 0 <= i < len(grid) and 0 <= j < len(grid[0]):
                list_3.append(grid[i][j])
            if i==row_index and j==col_index:
                list_3.remove(grid[i][j])
    return list_3

def sum_of_neighbors(grid):
    """[Uses the find_neighbors(row_index, col_index, grid) function to return 
    a new two dimensional nested list that depicts the
     sum of neighbors in each cell respectively.]
    list_1=representing rows
    list_2=representing columns
    Args:
        grid ([list]): [A two dimensional nested list]
    Returns:
        [list]: [A new two dimensional nested list]
    """
    list_1=[]
    for i in range(0,len(grid)):
        list_2=[]
        for j in range(0,len(grid[0])):
            sum_1=0
            l=find_neighbors(i,j,grid)
            for f in range(len(l)):
                sum_1+=int(l[f])
            list_2.append(sum_1)
        list_1.append(list_2)
    return list_1

def main():
    """[Create a grid with the given data from a file
    Display the grid
    Calculate the sum of all neighbors for each cell in the grid
    Display the new sum_of_neighbors grid]
    """
    with open('data_1.txt','r') as d_1,open('data_2.txt','r') as d_2:
        m=create_grid(d_1)
        l=create_grid(d_2) 
        display_grid(m) 
        display_grid(l)  
        d=sum_of_neighbors(m)
        e=sum_of_neighbors(l)
        display_grid(d)
        display_grid(e)

main()