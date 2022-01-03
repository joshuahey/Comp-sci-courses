import os
os.chdir(os.path.dirname(__file__))

def readAccounts(infile):
    """[read account file]

    Args:
        infile ([file]): [filemame]

    Returns:
        [dict]: [dictionary with account details]
    """
    dict1={}
    for line in infile:
        currentline=line.strip().split(">")
        try:
            currentline[1]=float(currentline[1])
            dict1[currentline[0]]=currentline[1]
        except ValueError:
            print("Warning! Account for",currentline[0]," not added: illegal value for balance")
    return dict1
 
def processAccounts(accounts):
    """[accounts processing]

    Args:
        accounts ([dict]): [dictionary with account details]
    """
    while 1:
        try:
            choice=input("Enter account name, or 'Stop' to exit:")
            if choice=="Stop":
                print("Exiting program...goodbye.")
                quit()
            accounts[choice]
            for j in accounts:
                if choice==j:
                    try:
                        i=float(input("Enter transaction amount for:"))
                        accounts[choice]+=i
                        print(accounts[choice])
                    except ValueError:
                        print("Warning! Incorrect amount. Transaction cancelled.")
        
        except KeyError:
            print("Warning! Account for ",choice ,"does not exist. Transaction cancelled. ")
        
def main():
    try:
        file_name=input("Enter Filename:")
        with open(file_name,"r") as f1:
            processAccounts(readAccounts(f1))
    except OSError:
        print("File error:",file_name,"does not exist" )
        print("Exiting program...goodbye.")

main()