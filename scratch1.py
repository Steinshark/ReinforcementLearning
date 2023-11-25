def pad_string(characters,direction,amount):

    if len(characters) > amount:
        return characters
        
    amount = int(amount)
    amount  = amount - len(characters)
    if direction == 'left':
 #       print("'",end='')
        for i in range(amount):
            print(f"{i}",end='')
            print(" ", end='')
        print(f"{characters}")

    if direction == 'right':
  #      for i in range(amount):
  #          print("'",end='')
        print(characters + " " * amount, end='')

def pad_left(characters,amount):
    pad_string(characters,'left',amount)

def pad_right(characters,amount):
    pad_string(characters,'right',amount)



a = ["Add","Delete","List"]
b   = ["Add to List","Delete information","List information"]

for command,descr in zip(a,b):  
    
    #Print command
    if len(command) > 10:
        print(command,end='')
    else:
        pad_right(command,10)

    #Print description
    if len(descr) > 10:
        print(descr,end='')
    else:
        pad_left(descr,10)

    #Print newline
    print()                         